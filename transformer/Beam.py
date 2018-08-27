""" Manage beam search info structure.

    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import numpy as np
import transformer.Constants as Constants

class Beam(object):
    # 对单个句子进行的beam_search
    ''' Store the neccesary info for beam search. '''
    # 维护每一步选的beam个句子中的最后一个词及其对应的上一步beam个句子中的哪个句子

    def __init__(self,beam_size,cuda=False):

        self.beam_size = beam_size
        self.done = False # 检查是否结束

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(beam_size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(beam_size).fill_(Constants.PAD)]
        self.next_ys[0][0] = Constants.BOS

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_lk):
        "Update the status and check for finished or not."
        num_words = word_lk.size(1)  #word_lk.size=[beam, n_vocab], 是每个单词的可能概率
        #之前的句子不同，这一步得到的单词概率也不同，所以之前存下的beam个句子是分开的

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)  # 所有的组合：size=[beam, n_vocab]
        else:
            beam_lk = word_lk[0]

        flat_beam_lk = beam_lk.view(-1) #铺平成1维，方便排序 beam*n_vocab,
        
        # 排序选出前beam_size个分数, best_scores_id.size=(beam,)
        best_scores, best_scores_id = flat_beam_lk.topk(self.beam_size,0,True,True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.beam_size,0,True,True) # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words  # 前一个部分的标号（是之前beam个句子中的哪一个）,size=(beam,)
        self.prev_ks.append(prev_k)  #size=[seq_len, beam]
        self.next_ys.append(best_scores_id - prev_k * num_words) # 当前选出单词的标号（是当前n_vocab个单词中的哪一个）, size=[seq_len, beam]

        # End condition is when top-of-beam is EOS.（最可能的单词是EOS）
        if self.next_ys[-1][0] == Constants.EOS:
            self.done = True
            self.all_scores.append(self.scores)

        return self.done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys] #从最后一次排序后的句子标号开始依次往前找之前选中的单词，构成翻译句子
            hyps = [[Constants.BOS] + h for h in hyps]
            dec_seq = torch.from_numpy(np.array(hyps))

        return dec_seq # (beam,seq_len)

    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis.

        Parameters.

             * `k` - the position in the beam to construct.

         Returns.

            1. The hypothesis
            2. The attention at each time step.
        """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1): #倒着计数
            hyp.append(self.next_ys[j+1][k]) #next_ys第一项全是BOS
            k = self.prev_ks[j][k]

        return hyp[::-1]
