
import numpy as np

valid_src_path=r'C:\Users\Zhihan Zhang\PycharmProjects\transformer-pytorch\transformer\train_source.txt'
valid_tgt_path=r'C:\Users\Zhihan Zhang\PycharmProjects\transformer-pytorch\transformer\train_target.txt'
test_src_path=r'C:\Users\Zhihan Zhang\PycharmProjects\transformer-pytorch\transformer\train_source_10.txt'
test_tgt_path=r'C:\Users\Zhihan Zhang\PycharmProjects\transformer-pytorch\transformer\train_target_10.txt'
valid_src=open(valid_src_path,'r',encoding='utf-8')
valid_tgt=open(valid_tgt_path,'r',encoding='utf-8')
test_src=open(test_src_path,'w',encoding='utf-8')
test_tgt=open(test_tgt_path,'w',encoding='utf-8')

cnt=83436 # total lines count

samples=[]
for _ in range(10):
    samples.append(int(np.random.rand()*cnt))

sample_src=[]
sample_tgt=[]
for i in range(cnt):
    src_line=valid_src.readline()
    tgt_line=valid_tgt.readline()
    if i in samples:
        sample_src.append(src_line)
        sample_tgt.append(tgt_line)

for string in sample_src:
    print(string,file=test_src,end='')
for string in sample_tgt:
    print(string,file=test_tgt,end='')

