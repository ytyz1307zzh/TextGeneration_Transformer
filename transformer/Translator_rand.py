''' This module will handle the text generation with top_k random sample search. '''

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from transformer.Models import Transformer
import transformer.Constants as Constants

class Translator_rand(object):
    ''' Load with trained model and handle the top_k random sample search '''

    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            emb_path=model_opt.emb_path,
            proj_share_weight=model_opt.proj_share_weight,
            embs_share_weight=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner_hid=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)

        prob_projection = nn.LogSoftmax()

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        if opt.cuda:
            model.cuda()
            prob_projection.cuda()
        else:
            model.cpu()
            prob_projection.cpu()

        model.prob_projection = prob_projection

        self.model = model
        self.model.eval()

    def translate_batch(self, src_batch):
        ''' Translation work in one batch '''

        # Batch size is in different location depending on data.
        src_seq, src_pos = src_batch # src_seq: batch x seq_len
        batch_size = src_seq.size(0)
        top_k = self.opt.top_k

        #- Encode
        enc_output, *_ = self.model.encoder(src_seq, src_pos) # batch x seq_len x d_model

        #--- Prepare hypotheses
        hypos = [[Constants.BOS] for _ in range(batch_size)] #每个inst建一个hypothesis list
        done = [False for _ in range(batch_size)] # batch内的每个句子又没有翻译完
        #最终的输出数据都保存在hypos里边，src_seq和enc_output都会动态裁剪
        hypo_inst_idx_map = {
            hypo_idx: inst_idx for inst_idx, hypo_idx in enumerate(range(batch_size))}
        #hypo-inst的索引字典, 所有通过hypo_idx对整个batch的索引都要先转化为inst_idx
        n_remaining_sents = batch_size #还剩几个句子没翻译

        #- Decode
        for i in range(self.model_opt.max_token_seq_len):

            len_dec_seq = i + 1

            # -- Preparing decoded data seq -- #
            # size: batch x seq
            dec_partial_seq = self.tt.LongTensor([hypos[i] for i in range(batch_size) if not done[i]])
            # wrap into a Variable (untrainable)
            dec_partial_seq = Variable(dec_partial_seq, volatile=True)

            # -- Preparing decoded pos seq -- #
            # size: 1 x seq
            dec_partial_pos = torch.arange(1, len_dec_seq + 1).unsqueeze(0)
            # size: batch x seq
            dec_partial_pos = dec_partial_pos.repeat(n_remaining_sents, 1)
            # 建造出[remaining_sents, len_dec_seq]的tensor，只需要考虑剩余未翻译完的句子
            # wrap into a Variable
            dec_partial_pos = Variable(dec_partial_pos.type(torch.LongTensor), volatile=True)

            if self.opt.cuda:
                dec_partial_seq = dec_partial_seq.cuda()
                dec_partial_pos = dec_partial_pos.cuda()

            # -- Decoding -- #
            dec_output, *_ = self.model.decoder(
                dec_partial_seq, dec_partial_pos, src_seq, enc_output) # batch x seq_len x d_model,只要seq_len的最后一步
            dec_output = dec_output[:, -1, :] # batch x d_model
            dec_output = self.model.tgt_word_proj(dec_output) # batch x n_vocab
            out = self.model.prob_projection(dec_output) # batch x n_vocab

            # batch x n_vocab
            word_probs = out.view(n_remaining_sents, -1).contiguous()

            def choose_word(word_prob,top_k): # size: n_vocab,
                # 排序选出前top_k个分数
                topk_scores, topk_scores_id = word_prob.topk(top_k, sorted=False) # select words with top k probabilities
                choice=topk_scores_id[np.random.randint(0,top_k)]
                return choice

            active_hypo_idx_list = []
            for hypo_idx in range(batch_size): # hypo_idx: 原batch中句子的索引, inst_idx: 剩余句子中的索引
                if done[hypo_idx]: #已经翻译完的
                    continue

                inst_idx = hypo_inst_idx_map[hypo_idx]
                chosen_word_id=choose_word(word_probs.data[inst_idx],top_k)
                hypos[hypo_idx].append(chosen_word_id) # 填入本次选择的单词

                if chosen_word_id == Constants.EOS: #下一个就是EOS，你这次也翻译完了
                    done[hypo_idx]=True
                else:
                    active_hypo_idx_list += [hypo_idx] #把没翻译完的句子对应的hypo_index构建列表


            if not active_hypo_idx_list:
                # all instances have finished their path to <EOS>
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_inst_idxs = self.tt.LongTensor(
                [hypo_inst_idx_map[k] for k in active_hypo_idx_list])

            # update the idx mapping
            hypo_inst_idx_map = {
                hypo_idx: inst_idx for inst_idx, hypo_idx in enumerate(active_hypo_idx_list)}

            def update_active_seq(seq_var, active_inst_idxs): # seq_var.size=[n_remain, seq_len]
                ''' Remove the src sequence of finished instances in one batch. '''

                # select the active instances in batch
                original_seq_data = seq_var.data.view(n_remaining_sents, -1) # [n_remain, seq_len]
                active_seq_data = original_seq_data.index_select(dim=0, index=active_inst_idxs) # [n_active, seq_len]

                return Variable(active_seq_data, volatile=True)

            def update_active_enc_info(enc_info_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''

                # select the active instances in batch
                original_enc_info_data = enc_info_var.data.view(
                    n_remaining_sents, -1, self.model_opt.d_model) # n_remain x seq_len x d_model
                active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs) # n_active x seq_len x d_model

                return Variable(active_enc_info_data, volatile=True)

            src_seq = update_active_seq(src_seq, active_inst_idxs)
            enc_output = update_active_enc_info(enc_output, active_inst_idxs)

            #- update the remaining size
            n_remaining_sents = len(active_inst_idxs)

        #- Return useful information
        return hypos # [batch, seq_len]
