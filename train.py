"""
This script handling the training process.
"""
#修改了batch_size, epoch, d_k, d_v, n_head和d_model的默认值
import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from DataLoader import DataLoader

def get_performance(crit, pred, gold, smoothing=False, num_class=None):
    """ Apply label smoothing if needed """
    #gold.size=[batch, seq_len], pred.size=[batch*seq_len, n_vocab]

    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
        raise NotImplementedError

    loss = crit(pred, gold.contiguous().view(-1)) #gold处理成1维

    pred = pred.max(1)[1] # max(1)求所有单词概率中的最大值，[1]取最大值下标（单词索引）

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)  
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum() #正确的单词个数，被pad的部分不能算

    return loss, n_correct

def train_epoch(model, training_data, crit, optimizer):
    """ Epoch operation in training phase"""

    model.train()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src, tgt = batch # src[0]=data,src[1]=pos, tgt同理
        gold = tgt[0][:, 1:]  # gold是正确答案，除掉BOS，size=[batch, seq_len]

        # forward
        optimizer.zero_grad()
        pred = model(src, tgt) #pred是model前向传播得到的结果, size=[batch*seq_len, n_vocab]

        # backward
        loss, n_correct = get_performance(crit, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate() #调整学习率

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()  #总词数
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.item()

    return total_loss/n_total_words, n_total_correct/n_total_words

def eval_epoch(model, validation_data, crit):
    """ Epoch operation in evaluation phase """

    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):

        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        pred = model(src, tgt)
        loss, n_correct = get_performance(crit, pred, gold) #只前向+评估没有后向

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.item()

    return total_loss/n_total_words, n_total_correct/n_total_words

def train(model, training_data, validation_data, crit, optimizer, opt):
    """ Start training """

    log_train_file = None
    log_valid_file = None

    if opt.log:   #日志记录
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_ppls = []
    print('[Info] Start training...')
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        #train process
        train_loss, train_accu = train_epoch(model, training_data, crit, optimizer)

        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        #validation process
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, crit)
        valid_ppl=math.exp(min(valid_loss, 100))
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=valid_ppl, accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_ppls += [valid_ppl]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':  #保存所有模型
                model_name = opt.save_model + '_ppl_{ppl:8.5f}.chkpt'.format(ppl=valid_ppl)
                torch.save(checkpoint, model_name)  #保存settings和model
            elif opt.save_mode == 'best':  #保存最佳模型
                model_name = opt.save_model + '_ppl_{ppl:8.5f}.chkpt'.format(ppl=valid_ppl)
                if valid_ppl <= min(valid_ppls):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))
    print('[Info] Finished.')

def main():
    """ Main function """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-emb_path', default=None)
    parser.add_argument('-trained_model', default=None)
    parser.add_argument('-current_step', type=int, default=0)

    parser.add_argument('-epoch', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=32)

    #parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-d_model', type=int, default=300)
    parser.add_argument('-d_inner_hid', type=int, default=500)
    parser.add_argument('-d_k', type=int, default=50)
    parser.add_argument('-d_v', type=int, default=50)

    parser.add_argument('-n_head', type=int, default=6)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None) # 日志保存地址(without suffix)
    parser.add_argument('-save_model', default=None) # 模型保存地址(without suffix)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    #========= Preparing DataLoader =========#
    training_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'],
        batch_size=opt.batch_size,
        cuda=opt.cuda)

    validation_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['valid']['src'],
        tgt_insts=data['valid']['tgt'],
        batch_size=opt.batch_size,
        shuffle=False,
        test=True,
        cuda=opt.cuda)

    opt.src_vocab_size = training_data.src_vocab_size
    opt.tgt_vocab_size = training_data.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight and training_data.src_word2idx != training_data.tgt_word2idx:
        print('[Warning]',
              'The src/tgt word2idx table are different but asked to share word embedding.')

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        emb_path=opt.emb_path,
        proj_share_weight=opt.proj_share_weight,
        embs_share_weight=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)

    if opt.trained_model:
        checkpoint = torch.load(opt.trained_model)
        transformer.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

    #print(transformer)

    optimizer = ScheduledOptim(
        optim.Adam(
            transformer.get_trainable_parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps, opt.current_step)


    def get_criterion(vocab_size):
        """ With PAD token zero weight """
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0  #被pad的部分在计算loss的时候权重设为0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit = get_criterion(training_data.tgt_vocab_size)

    if opt.cuda:
        transformer = transformer.cuda()
        crit = crit.cuda()

    train(transformer, training_data, validation_data, crit, optimizer, opt)

if __name__ == '__main__':
    main()
