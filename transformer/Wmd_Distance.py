from pyemd import emd
import numpy as np
stopwords_list=[45, 289, 196, 3265, 57, 166, 12170, 5044, 85, 396, 16406, 4965, 34984, 22, 107, 30, 674, 71, 75, 16738, 2942,
                24, 51, 1007, 43, 105, 48, 13409, 1004, 106, 46, 42, 1235, 41, 16, 162, 159, 917, 18, 36, 19, 39, 34, 55, 138,
                37, 35, 44, 522, 92, 264, 123, 918, 11, 33, 4, 9, 38, 87, 50, 117, 23, 211, 114, 7, 26, 25, 14, 21, 63, 102,
                122, 79, 135, 109, 110, 53, 1073, 1272, 8, 29, 64, 139, 10, 70, 17, 142, 78, 128, 382, 493, 131, 446, 191,
                67, 65, 115, 742, 201, 68, 134, 154, 240, 310, 60, 100, 72, 81, 129, 88, 2098, 40, 95, 265, 219, 104, 77,
                321, 195, 1538, 2163, 90, 47, 124, 3320, 193, 118, 1972, 31723, 1997, 4872, 1769, 37142, 3528, 25225, 105188,
                163655, 73334, 66023, 178189, 6018, 75364, 4209, 14106, 185139, 128077, 239580, 231, 188794,0,1,2,3,6]
def Wmd_Distance(src_seq, cur_sent,embed_mat):

    #src_seq=[index for index in src_seq if index not in stopwords_list]
    #cur_sent=[index for index in cur_sent if index not in stopwords_list]

    word_set=list(set(src_seq + cur_sent))
    vocab_len = len(word_set)
    vocab=[word for word in word_set]
    # 计算词之间的语义距离
    distance_matrix = np.zeros((vocab_len,vocab_len))
    for r in range(vocab_len):
        for c in range(vocab_len):
            distance_matrix[r][c] = compute_distance(vocab[r],vocab[c],embed_mat)
    # 计算归一化的词频概率
    d1 = compute_normalized_word_freq(src_seq,word_set)
    d2 = compute_normalized_word_freq(cur_sent,word_set)

    # 计算词移距离
    wmd_distance = emd(d1, d2, distance_matrix) # (0,1)内的值，越小越好
    return wmd_distance - 1

def compute_distance(word1,word2,embed_mat):

    word_embed1=embed_mat[word1]
    word_embed2=embed_mat[word2]

    def length(vector):
        result=sum(vector*vector)
        result=np.sqrt(result)
        return result

    distance = 1 - sum(word_embed1*word_embed2)/(length(word_embed1)*length(word_embed2))
    return distance

def compute_normalized_word_freq(sent, word_set):
    word_freq=np.zeros(len(word_set))
    for word in sent:
        ind=word_set.index(word)
        word_freq[ind]+=1

    word_freq/=len(sent)
    return word_freq
'''
embed_mat=np.array([[0,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8]])
src_seq=[2,3,2]
cur_sent=[1,2,4,0,3,2,1,2,3]
print(Wmd_Distance(src_seq,cur_sent,embed_mat))
'''

