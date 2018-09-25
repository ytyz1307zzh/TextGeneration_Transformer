import numpy as np

def word_embedding_init(emb_path):
    emb_file=open(emb_path,'r',encoding='utf-8')
    emb_mat=[]
    cnt=0
    for line in emb_file:
        cnt+=1
        if cnt>10000:
            break
        emb_mat.append(line.strip().split())

    embed_mat=np.array(emb_mat)
    embed_words=embed_mat[:,:1].squeeze()
    embed_paras=embed_mat[:,1:].astype(np.float)

    return {word:para for word,para in zip(embed_words,embed_paras)}

def cal_bow(sent,embed_mat):
    sent_vec=np.zeros(50)
    for word in sent:
        sent_vec+=embed_mat[word]

    sent_vec/=len(sent)

    return sent_vec

emb_path=r'E:\NLP\glove.6B.50d.txt'
embed_mat=word_embedding_init(emb_path)
'''
text_path='./bow.txt'
text_file=open(text_path,'r',encoding='utf-8')

sent1=text_file.readline()
sent2=text_file.readline()
'''
sent1='can you introduce yourself'
sent2='what is your name'

sent1=sent1.strip().split()
sent2=sent2.strip().split()
sent1_vec=cal_bow(sent1,embed_mat)
sent2_vec=cal_bow(sent2,embed_mat)

def length(vector):
    result=sum(vector*vector)
    result=np.sqrt(result)
    return result

relevant_score=sum(sent1_vec*sent2_vec)/(length(sent1_vec)*length(sent2_vec))
print(relevant_score)

from pyemd import emd
vocab_len = len(set(sent1 + sent2))
# 计算词之间的语义距离
distance_matrix = compute_cosine_between_token(doc1, doc2)
distance_matrix.shape = (vocab_len, vocab_len)
# 计算归一化的词频概率
d1 = compute_normalized_word_freq(doc1)
d2 = compute_normalized_word_freq(doc2)
d1.shape = d2.shape = (vocab_len)
# 计算词移距离
wmd = emd(d1, d2, distance_matix)


