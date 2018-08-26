''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer.Models import Transformer
from transformer.Diverse_Beam import Diverse_Beam
import transformer.Constants as Constants

class Translator_idbs(object):
    ''' Load with trained model and handle the beam search '''

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
        src_seq, src_pos = src_batch
        batch_size = src_seq.size(0)
        beam_size = self.opt.beam_size

        src_seq=src_seq[:, 1:-1]
        src_pos=src_pos[:, 1:-1]

        #- Encode
        enc_output, *_ = self.model.encoder(src_seq, src_pos)

        #--- Repeat data for beam
        src_seq = Variable(
            src_seq.data.repeat(1, beam_size).view(
                src_seq.size(0) * beam_size, src_seq.size(1)))  #(batch x beam) x seq_len

        enc_output = Variable(
            enc_output.data.repeat(1, beam_size, 1).view(
                enc_output.size(0) * beam_size, enc_output.size(1), enc_output.size(2)))

        #--- Prepare beams
        beams = [Diverse_Beam(beam_size, self.opt.cuda) for _ in range(batch_size)] #每个inst建一个beam类
        #最终的输出数据都保存在beams里边，src_seq和enc_output都会动态裁剪
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))}  #beam-inst的索引字典
        n_remaining_sents = batch_size #还剩几个句子没翻译

        #- Decode
        for i in range(self.model_opt.max_token_seq_len):

            len_dec_seq = i + 1

            # -- Preparing decoded data seq -- #
            # size: batch x beam x seq
            bos = self.tt.LongTensor(batch_size,beam_size).fill_(Constants.BOS).unsqueeze(-1)
            dec_partial_seq = torch.stack([
                b.get_current_state() for b in beams if not b.done])
            dec_partial_seq = torch.cat([bos,dec_partial_seq],dim=-1)
            # size: (batch * beam) x seq
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            # wrap into a Variable (untrainable)
            dec_partial_seq = Variable(dec_partial_seq, volatile=True)

            # -- Preparing decoded pos seq -- #
            # size: 1 x seq
            dec_partial_pos = torch.arange(1, len_dec_seq + 1).unsqueeze(0)
            # size: (batch * beam) x seq
            dec_partial_pos = dec_partial_pos.repeat(n_remaining_sents * beam_size, 1)
            # 建造出[remaining_sents*beam, len_dec_seq]的tensor，只需要考虑剩余未翻译完的句子
            # wrap into a Variable
            dec_partial_pos = Variable(dec_partial_pos.type(torch.LongTensor), volatile=True)

            if self.opt.cuda:
                dec_partial_seq = dec_partial_seq.cuda()
                dec_partial_pos = dec_partial_pos.cuda()

            # -- Decoding -- #
            dec_output, *_ = self.model.decoder(
                dec_partial_seq, dec_partial_pos, src_seq, enc_output) # (batch x beam) x seq_len x d_model,只要seq_len的最后一步
            dec_output = dec_output[:, -1, :] # (batch * beam) * d_model
            dec_output = self.model.tgt_word_proj(dec_output)
            out = self.model.prob_projection(dec_output)

            # batch x beam x n_words
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []
            for beam_idx in range(batch_size):
                if beams[beam_idx].done:
                    continue

                inst_idx = beam_inst_idx_map[beam_idx]
                if not beams[beam_idx].advance(word_lk.data[inst_idx]): #下一个就是EOS，也算你翻译完了
                    active_beam_idx_list += [beam_idx] #把没翻译完的句子对应的beam_index构建列表

            if not active_beam_idx_list:
                # all instances have finished their path to <EOS>
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_inst_idxs = self.tt.LongTensor(
                [beam_inst_idx_map[k] for k in active_beam_idx_list])

            # update the idx mapping
            beam_inst_idx_map = {
                beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

            def update_active_seq(seq_var, active_inst_idxs): # seq_var.size=[batch*beam, seq_len]
                ''' Remove the src sequence of finished instances in one batch. '''

                inst_idx_dim_size, *rest_dim_sizes = seq_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents # [n_active*beam]
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_seq_data = seq_var.data.view(n_remaining_sents, -1) # [batch, beam*seq_len]
                active_seq_data = original_seq_data.index_select(0, active_inst_idxs) # [n_active, beam*seq_len]
                active_seq_data = active_seq_data.view(*new_size) # [n_active*beam, seq_len]

                return Variable(active_seq_data, volatile=True)

            def update_active_enc_info(enc_info_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''

                inst_idx_dim_size, *rest_dim_sizes = enc_info_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_enc_info_data = enc_info_var.data.view(
                    n_remaining_sents, -1, self.model_opt.d_model)
                active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
                active_enc_info_data = active_enc_info_data.view(*new_size)

                return Variable(active_enc_info_data, volatile=True)

            src_seq = update_active_seq(src_seq, active_inst_idxs)
            enc_output = update_active_enc_info(enc_output, active_inst_idxs)

            #- update the remaining size
            n_remaining_sents = len(active_inst_idxs)

        #- Return useful information
        all_hyp = []

        for beam_idx in range(batch_size):
            hypo=[]
            for sent in beams[beam_idx].prev_sents:
                 hypo.extend(sent) # 将prev_sent转化成一维list
            all_hyp.append(hypo)

        return all_hyp # all_hyp.size()=[batch, seq_len]
