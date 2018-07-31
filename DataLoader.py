''' Data Loader class for training iteration '''
'''构建时输入src_insts，tgt_insts以及source和target的word2idx字典；
按batch进行迭代，每次迭代时挑选出下一个batch的instances并且做padding，同时构建position数据；
每次迭代生成(src_data, src_pos), (tgt_data, tgt_pos)的batch，四部分均为shape=[batch_size,max_seq_len]
'''
import random
import numpy as np
import torch
from torch.autograd import Variable
import transformer.Constants as Constants

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, src_word2idx, tgt_word2idx,
            src_insts=None, tgt_insts=None,
            cuda=True, batch_size=32, shuffle=True, test=False):
        #测试的时候没有tgt_insts

        assert src_insts
        assert len(src_insts) >= batch_size

        if tgt_insts:
            assert len(src_insts) == len(tgt_insts)

        self.cuda = cuda
        self.test = test
        self._n_batch = int(np.ceil(len(src_insts) / batch_size))

        self._batch_size = batch_size

        self._src_insts = src_insts
        self._tgt_insts = tgt_insts
        
        # 标号到单词的字典
        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}

        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word

        self._iter_count = 0 # 迭代batch数量统计

        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()
    
    #以下使用@property声明可引用属性
    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        #测试的时候没有target，所以需要判断_tgt_insts是不是None
        
        if self._tgt_insts:
            #训练的时候shuffle需要成对shuffle,先zip再unzip
            paired_insts = list(zip(self._src_insts, self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            random.shuffle(self._src_insts)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            #Pad the instance to the max seq length in batch 

            max_len = max(len(inst) for inst in insts) # 这个batch中的最长序列长度

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts]) #对insts中的每个句子进行padding，shape=(batch_size,max_seq_len)

            inst_position = np.array([
                [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data]) #对insts中的每个句子进行position标号，便于之后进行position embedding, shape=(batch_size,max_seq_len)

            # 构建句子tensor和位置tensor
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)
            inst_position_tensor = Variable(
                torch.LongTensor(inst_position), volatile=self.test)
            
            #应用GPU计算
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            #如果没全部迭代一遍，构建下一个batch
            batch_idx = self._iter_count #下一个batch的标号
            self._iter_count += 1
            
            start_idx = batch_idx * self._batch_size  #下一个batch的start
            end_idx = (batch_idx + 1) * self._batch_size  #下一个batch的end

            src_insts = self._src_insts[start_idx:end_idx]  #下一个batch
            src_data, src_pos = pad_to_longest(src_insts)  #对下一个batch进行padding

            if not self._tgt_insts:
                return src_data, src_pos
            else:
                tgt_insts = self._tgt_insts[start_idx:end_idx]  #如果是训练过程，对target对应的句子构建batch并padding
                tgt_data, tgt_pos = pad_to_longest(tgt_insts)
                return (src_data, src_pos), (tgt_data, tgt_pos)

        else:
            #已经迭代到头了，本次迭代结束，打乱数据集，迭代次数置零，准备下一次迭代
            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
