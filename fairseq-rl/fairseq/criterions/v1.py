import math
import torch.nn.functional as F
import collections
import torch
import numpy
from collections import defaultdict

from fairseq.sequence_generator import SequenceGenerator
from fairseq import utils, bleu

from . import FairseqCriterion, register_criterion

import csv

@register_criterion('v1')
class V1Criterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    # def forward(self, model, sample, reduce=True):
    #     """Compute the loss for the given sample.
    #
    #     Returns a tuple with three elements:
    #     1) the loss
    #     2) the sample size, which is used as the denominator for the gradient
    #     3) logging outputs to display while training
    #     """
    #     net_output = model(**sample['net_input'])
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     lprobs = lprobs.view(-1, lprobs.size(-1))
    #     target = model.get_targets(sample, net_output).view(-1)
    #     loss = F.nll_loss(lprobs, target, size_average=False,
    #                       ignore_index=self.padding_idx,
    #                       reduce=reduce)
    #     sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
    #     logging_output = {
    #         'loss': utils.item(loss.data) if reduce else loss.data,
    #         'ntokens': sample['ntokens'],
    #         'sample_size': sample_size,
    #     }
    #     return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        # sample mode
        #print('!!!RL loss.')
        model.eval()
        # src_dict = self.task.source_dictionary
        tgt_dict = self.task.target_dictionary
        eos_idx = self.task.target_dictionary.eos()
        sample_beam = 1
        translator = SequenceGenerator([model], tgt_dict=tgt_dict, sampling=self.args.multinomial_sample_train,
                                       beam_size=sample_beam, minlen=1)
        translator.cuda()
        ct = 0
        translations = []

        s = utils.move_to_cuda(sample)
        input = s['net_input']
        max_len = 200
        with torch.no_grad():
            hypos = translator.generate(
                input['src_tokens'],
                input['src_lengths'],
                beam_size=sample_beam,
                maxlen=max_len,
            )
        for i, id in enumerate(s['id'].data):
            src = input['src_tokens'].data[i, :]
            # remove padding from ref
            ref = utils.strip_pad(s['target'].data[i, :], tgt_dict.pad()) if s['target'] is not None else None
            translations.append((id, src, ref, hypos[i]))
            ct += 1
        # print("sample batch size:", ct)

        model.train()

        # MLE loss
        mle_net_output = model(**sample['net_input'])
        mle_lprobs = model.get_normalized_probs(mle_net_output, log_probs=True)
        mle_lprobs = mle_lprobs.view(-1, mle_lprobs.size(-1))
        mle_target = model.get_targets(sample, mle_net_output).view(-1)
        mle_loss = F.nll_loss(mle_lprobs, mle_target, size_average=False,
                              ignore_index=self.padding_idx, reduce=reduce)
        mle_tokens = sample['ntokens']
        avg_mle_loss = mle_loss / mle_tokens
        print('avg_mle_loss:', avg_mle_loss)
        # RL loss
        batch_rl_loss = 0
        batch_tokens = 0
        id = 0
        result = []
        for sample_id, src_tokens, tgt_tokens, hypos in translations:
            # calculate bleu
            id += 1
            hypo = hypos[0]  # only extract the first hypo (beam1 or sample1)
            trans_tokens = hypo['tokens']
            reward = self.compute_gleu(tgt_tokens.cpu(), trans_tokens.cpu(), max_order=self.args.max_order, gram=self.args.gram).cuda()
            result.append((id, reward.item(), tgt_tokens.size(0), trans_tokens.size(0)))
            # one_sample loss calculation
            tgt_input_tokens = trans_tokens.new(trans_tokens.shape).fill_(0)
            assert trans_tokens[-1] == eos_idx
            tgt_input_tokens[0] = eos_idx
            tgt_input_tokens[1:] = trans_tokens[:-1]
            train_sample = {
                'net_input': {
                    'src_tokens': src_tokens.view(1, -1),
                    'src_lengths': torch.LongTensor(src_tokens.numel()).view(1, -1),
                    'prev_output_tokens': tgt_input_tokens.view(1, -1),
                },
                'target': trans_tokens.view(1, -1)
            }
            train_sample = utils.move_to_cuda(train_sample)
            net_output = model(**train_sample['net_input'])
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(train_sample, net_output).view(-1, 1)
            non_pad_mask = target.ne(tgt_dict.pad())
            lprob = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
            rl_loss = torch.sum(lprob * reward)  # one sample loss
            ntokens = len(train_sample['target'])
            batch_tokens += ntokens
            batch_rl_loss += rl_loss

        avg_rl_loss = batch_rl_loss / batch_tokens
        print('avg_rl_loss:', avg_rl_loss)
        if self.args.mle_weight:
            assert self.args.rl_weight
            total_loss = self.args.mle_weight * avg_mle_loss + self.args.rl_weight * avg_rl_loss
            total_tokens = batch_tokens + mle_tokens
        else:
            total_loss = avg_rl_loss
            total_tokens = batch_tokens
        logging_output = {
            'loss': utils.item(total_loss.data),
            'ntokens': total_tokens,
            'sample_size': total_tokens,
        }
        print('total: ',total_loss)
        return total_loss, total_tokens, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output

    def _get_ngrams(self, segment, max_order):
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def compute_gleu(self, reference_corpus, translation_corpus, max_order=4, gram=0, smooth=False):
        scores = torch.zeros(max_order)
        reference_array = numpy.array(reference_corpus)
        translation_array = numpy.array(translation_corpus)
        matches_by_order = [0] * max_order
        possible_matches_by_order_ref = [0] * max_order
        possible_matches_by_order_trans = [0] * max_order
        reference_length = 0
        translation_length = 0
        reference_length += reference_array.shape[0]
        translation_length += translation_array.shape[0]
        merged_ref_ngram_counts = collections.Counter()
        merged_ref_ngram_counts |= self._get_ngrams(reference_array, max_order)
        translation_ngram_counts = self._get_ngrams(translation_array, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches_trans = translation_length - order + 1
            if possible_matches_trans > 0:
                possible_matches_by_order_trans[order-1] += possible_matches_trans
            possible_matches_ref = reference_length - order + 1
            if possible_matches_ref > 0:
                possible_matches_by_order_ref[order-1] += possible_matches_ref
        precisions = [0] * max_order
        recalls = [0] * max_order

        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order_trans[i] + 1.))
                recalls[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order_ref[i] + 1.))
            else:
                if possible_matches_by_order_trans[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order_trans[i])
                else:
                    precisions[i] = 0.0
                
                if possible_matches_by_order_ref[i] > 0:
                    recalls[i] = (float(matches_by_order[i]) / possible_matches_by_order_ref[i])
                else:
                    recalls[i] = 0.0
        for i in range(max_order):
            scores[i] = min(precisions[i],recalls[i])

        if self.args.modgleu:
            if reference_length < max_order and translation_length < max_order:
                order = max(reference_length, translation_length)
                scores = scores[0:order]
            else:
                order = max_order
        else:
            order = max_order
        
        if gram == 0:
            if min(scores) > 0:
                log_scores = torch.log(scores)
                p_log_sum = torch.sum((1. / order) * log_scores)
                geo_mean = torch.exp(p_log_sum)
                return geo_mean
            else:
                return 0.0
        else:
            if scores[gram] > 0:
                return scores[gram]
            else:
                return 0.0