""" Manage beam search info structure.

    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import numpy as np
import transformer.Constants as Constants
from transformer.Diversity import hamming_diversity, n_gram_diversity
from transformer.Wmd_Distance import Wmd_Distance

class Cos_Beam(object):
    # 对单个句子进行的beam_search
    ''' Store the neccesary info for beam search. '''
    # 维护每一步选的beam个句子中的最后一个词及其对应的上一步beam个句子中的哪个句子

    def __init__(self,beam_size,lambda_1=2/3,lambda_2=2/3,lambda_3=2/3,wmd_weight=10,cuda=False):

        self.beam_size = beam_size
        self.done = False # 检查是否结束
        self.Lambda_1=lambda_1 # diversity factor for hamming diversity
        self.Lambda_2=lambda_2 # diversity factor for bi-gram diversity
        self.Lambda_3=lambda_3 # diversity factor for tri-gram diversity
        self.wmd_weight=wmd_weight

        self.sent_cnt=0 # 已完成的句子个数
        self.prev_sents=[] # 已完成的句子, (sent, seq_len)

        self.tt = torch.cuda if cuda else torch
        self.cuda = cuda

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(beam_size).zero_() # size=(beam,)

        # The backpointers at each time-step.
        self.prev_ks = []
        self.next_ys = []
        self.init_lists()

    def init_lists(self):
        self.next_ys.clear()
        self.next_ys = [self.tt.LongTensor(self.beam_size).fill_(Constants.BOS)]
        self.prev_ks.clear() # 清空回溯列表
        self.scores = self.tt.FloatTensor(self.beam_size).zero_() #清空上一句的分数

    def get_current_state(self, prev_sents=True):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis(prev_sents)

    def advance(self, word_lk, embed_mat, src_seq): #src_seq.size=[beam_size,seq_len]
        "Update the status and check for finished or not."
        num_words = word_lk.size(1)  #word_lk.size=[beam, n_vocab], 是每个单词的可能概率
        #之前的句子不同，这一步得到的单词概率也不同，所以之前存下的beam个句子是分开的
        beam_size=self.beam_size

        diversity=torch.zeros(beam_size,num_words) #(beam, n_vocab)
        wmd_distance=torch.zeros(beam_size,num_words)

        if len(self.next_ys)>1:
            cur_sents=self.get_current_state(prev_sents=False).unsqueeze(1).repeat(1,num_words,1) # (beam, n_vocab, seq_len)
        word_ids=torch.arange(0,num_words,dtype=torch.int64)
        word_ids=word_ids.unsqueeze(0).unsqueeze(-1).expand(beam_size,num_words,1)
        cur_sents=torch.cat([cur_sents,word_ids],dim=-1) if len(self.next_ys)>1 else word_ids# 将每个可能单词放入cur_sent计算各自的diversity

        #compute wmd distance
        for beam_id in range(beam_size):
            for sent_id in range(round(num_words*0.1)):
                wmd_distance[beam_id][sent_id]+=self.wmd_weight*Wmd_Distance(src_seq[beam_id],cur_sents[beam_id][sent_id],embed_mat)

        '''
        if self.sent_cnt>0: # 对第二个及之后的句子，加入hamming diversity
            #compute diversity
            for beam_id in range(beam_size):
                for sent_id in range(round(num_words*0.1)):
                    for prev_sent in self.prev_sents: # 和之前的每个句子比较，diversity求和
                        diversity[beam_id][sent_id]+=self.Lambda_1*hamming_diversity(prev_sent,cur_sents[beam_id][sent_id])
                        diversity[beam_id][sent_id]+=self.Lambda_2*n_gram_diversity(prev_sent,cur_sents[beam_id][sent_id],n=2)
                        diversity[beam_id][sent_id]+=self.Lambda_3*n_gram_diversity(prev_sent,cur_sents[beam_id][sent_id],n=3)
         '''
        #print('diversity:',diversity)

        if self.cuda:
            diversity=diversity.cuda()

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk) + diversity - wmd_distance  # 所有的组合：size=[beam, n_vocab]
        else:
            beam_lk = word_lk[0] + diversity[0] - wmd_distance[0] # t=0, 直接找beam_size个概率最大的单词

        flat_beam_lk = beam_lk.view(-1) #铺平成1维，方便排序 beam*n_vocab,

        # 排序选出前beam_size个分数, best_scores_id.size=(beam,)
        best_scores, best_scores_id = flat_beam_lk.topk(self.beam_size+1,0,True,True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.beam_size+1,0,True,True) # 2nd sort

        best_scores=best_scores.tolist()
        best_scores_id=best_scores_id.tolist()
        try:
            quote_id=best_scores_id.index(Constants.QUOTATION_MARK) # 如果出现了引号
            best_scores=[best_scores[id] for id in range(len(best_scores)) if id!=quote_id]
            best_scores_id=[best_scores_id[id] for id in range(len(best_scores_id)) if id!=quote_id]
        except ValueError:
            best_scores=best_scores[:-1]
            best_scores_id=best_scores_id[:-1]
        best_scores=self.tt.FloatTensor(best_scores)
        best_scores_id=self.tt.LongTensor(best_scores_id)

        self.scores = best_scores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words  # 前一个部分的标号（是之前beam个句子中的哪一个）,size=(beam,)
        self.prev_ks.append(prev_k)  #size=[seq_len, beam]
        self.next_ys.append(best_scores_id - prev_k * num_words) # 当前选出单词的标号（是当前n_vocab个单词中的哪一个）, size=[seq_len, beam]

        # End condition is when top-of-beam is EOS.（最可能的单词是EOS）
        if self.next_ys[-1][0] == Constants.EOS:
            self.done = True
            self.end_of_sentence()

        elif self.next_ys[-1][0] == Constants.PERIOD\
                or self.next_ys[-1][0] == Constants.QUESTION_MARK\
                or self.next_ys[-1][0] == Constants.EXCLAMATION_MARK: # 最可能的单词是.?!说明一个句子结束了
            self.end_of_sentence()

        return self.done

    def end_of_sentence(self):
        dec_seq=self.get_current_state(prev_sents=False)
        self.prev_sents.append(dec_seq[0].tolist()) # 最佳答案作为这一句的DBS结果
        self.sent_cnt+=1
        self.init_lists()

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_tentative_hypothesis(self, prev_sents=True):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1 and self.sent_cnt==0:
            dec_seq = self.next_ys[0].unsqueeze(1)

        elif len(self.next_ys) == 1 and self.sent_cnt>0 and prev_sents:
            prev=[]
            for sent in self.prev_sents:
                prev.extend(sent)  # 将prev_sent转化成一维list
            hyps = self.tt.LongTensor(prev).unsqueeze(0).repeat(self.beam_size,1) # (beam, seq_len)
            dec_seq = torch.from_numpy(np.array(hyps))

        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys] #从最后一次排序后的句子标号开始依次往前找之前选中的单词，构成翻译句子

            if self.sent_cnt>0 and prev_sents:
                prev=[]
                for sent in self.prev_sents:
                    prev.extend(sent) # 将prev_sent转化成一维list
                hyps = torch.cat([self.tt.LongTensor(prev).unsqueeze(0).repeat(self.beam_size,1), self.tt.LongTensor(hyps)],dim=-1)
                # (beam, seq_len)

            dec_seq = torch.from_numpy(np.array(hyps))
        if self.cuda:
            dec_seq=dec_seq.cuda()

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

        return hyp[::-1] # 前边得到的句子是反的
