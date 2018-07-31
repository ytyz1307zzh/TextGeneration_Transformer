""" Define the Transformer model """
#去掉了encoder的position encoding, 去掉了source的bos和eos
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Modules import BottleLinear as Linear
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

def position_encoding_init(n_position, d_pos_vec):
    """ Init the sinusoid position encoding table """
    #n_position=max_seq_len+1, d_pos_vec=d_model

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor) # seq_len x d_model

def word_embedding_init(emb_path):
    embed_mat=np.loadtxt(emb_path,encoding='utf-8',dtype=np.str)
    embed_mat=embed_mat[:,1:].astype(np.float)

    const_embed=np.random.rand(4,300)*2-1 # bos,eos,pad,unk
    embed_mat=np.row_stack((const_embed,embed_mat))

    return torch.from_numpy(embed_mat).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    """ Indicate the padding-related part to mask """
    assert seq_q.dim() == 2 and seq_k.dim() == 2 # batch_size x seq_len
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    # mask的key对每一个query都是mask的，所以增加一维后直接expand
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # b x 1 x sk, 转化为0-1矩阵
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # b x sq x sk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    """ Get an attention mask to avoid using the subsequent info."""
    assert seq.dim() == 2 # batch_size x seq_len
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # 对于每一个query中的元素，比他下标大的key中的元素都mask掉
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, n_src_vocab, n_max_seq, emb_path=None, n_layers=6, n_head=6, d_k=50, d_v=50,
            d_word_vec=300, d_model=300, d_inner_hid=500, dropout=0.1):

        super(Encoder, self).__init__()

        self.n_max_seq = n_max_seq
        self.d_model = d_model

        assert n_src_vocab==400004 and d_word_vec==300
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.src_word_emb.weight.data = word_embedding_init(emb_path)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False): #src_seq,src_pos:[batch,max_seq_len]
        # Word embedding look up
        enc_input = self.src_word_emb(src_seq)

        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,

class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """
    def __init__(
            self, n_tgt_vocab, n_max_seq, emb_path=None, n_layers=6, n_head=6, d_k=50, d_v=50,
            d_word_vec=300, d_model=300, d_inner_hid=500, dropout=0.1):

        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        assert n_tgt_vocab==400004 and d_word_vec==300
        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.tgt_word_emb.weight.data = word_embedding_init(emb_path)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        # Word embedding look up
        dec_input = self.tgt_word_emb(tgt_seq)

        # Position Encoding addition
        dec_input += self.position_enc(tgt_pos)

        # decoder的self-attention包含padding的mask和顺序屏蔽的mask
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        
        # encoder-decoder attention只包含padding的mask
        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq, emb_path=None, n_layers=6, n_head=6,
            d_word_vec=300, d_model=300, d_inner_hid=500, d_k=50, d_v=50,
            dropout=0.1, proj_share_weight=True, embs_share_weight=True):

        super(Transformer, self).__init__()
        self.encoder = Encoder(
            n_src_vocab, n_max_seq, emb_path=emb_path, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder = Decoder(
            n_tgt_vocab, n_max_seq, emb_path=emb_path, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=False) # d_model到n_tgt_vocab的映射，最终预测结果
        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'

        if proj_share_weight:
            # Share the weight matrix between tgt word embedding/projection
            assert d_model == d_word_vec
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight

        if embs_share_weight:
            # Share the weight matrix between src/tgt word embeddings
            # assume the src/tgt word vec size are the same
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def get_trainable_parameters(self):
        #Avoid updating the position encoding
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        #去掉source的bos和eos，转成纯粹的词袋模型
        src_seq=src_seq[:, 1:-1]
        src_pos=src_pos[:, 1:-1]
        
        #decoder的输入不需要EOS(shift right)
        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]
        
        # decoder每一层的encoder-decoder attention中K和V都来源于encoder的最终输出
        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output) 
        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))
