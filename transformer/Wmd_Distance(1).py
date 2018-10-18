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

    src_emb=np.array([embed_mat[word] for word in src_seq])
    cur_emb=np.array([embed_mat[word] for word in cur_sent])

    src_bow=np.mean(src_emb,axis=0)
    cur_bow=np.mean(cur_emb,axis=0)

    def length(vector):
        result=np.sum(vector*vector)
        result=np.sqrt(result)
        return result

    distance = -1 - np.sum(src_bow*cur_bow)/(length(src_bow)*length(cur_bow))# 后面这一项是[-1,1]内的值，越大越好

    return distance
'''
embed_mat=np.array([[0,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8]])
src_seq=[2,3,2]
cur_sent=[1,2,4,0,3,2,1,2,3]
distance=Wmd_Distance(src_seq,cur_sent,embed_mat)
print(distance)
print(distance.dtype)
'''


