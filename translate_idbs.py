''' Translate input text with trained model. '''

import torch
import argparse
from tqdm import tqdm
from transformer.Translator_idbs import Translator_idbs
from DataLoader import DataLoader
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='preprocess file to provide vocabulary')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-lambda_1',type=float, default=2/3,
                        help='diversity factor for hamming diversity')
    parser.add_argument('-lambda_2',type=float, default=2/3,
                        help='diversity factor for bi-gram diversity')
    parser.add_argument('-lambda_3',type=float, default=2/3,
                        help='diversity factor for tri-gram diversity')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']

    test_src_word_insts = read_instances_from_file(
        opt.src,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)

    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])

    test_data = DataLoader(
        preprocess_data['dict']['src'],
        preprocess_data['dict']['tgt'],
        src_insts=test_src_insts,
        cuda=opt.cuda,
        shuffle=False,
        batch_size=opt.batch_size)

    translator = Translator_idbs(opt)
    translator.model.eval()

    print('[Info] Start translating...')
    f=open(opt.output, 'w')
    for batch in tqdm(test_data, mininterval=2, desc='  - (Test)', leave=False):
        all_hyp= translator.translate_batch(batch)
        for idx_seq in all_hyp:
            pred_line = ' '.join([test_data.tgt_idx2word[idx] for idx in idx_seq])  #转化成单词拼接起来
            f.write(pred_line + '\n')
            f.flush()
    f.close()
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
