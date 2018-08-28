from nltk.util import ngrams

def hamming_diversity(prev_sent, cur_sent):
    """Return the Hamming distance between equal-length sequences"""
    div=0
    for i in range(min(len(prev_sent),len(cur_sent))):
        if prev_sent[i]!=cur_sent[i]:
            div+=1
    div+=abs(len(prev_sent)-len(cur_sent)) # 多出来的也算作diversity
    return div

def n_gram_diversity(prev_sent, cur_sent, n): # n: n-gram parameter "n"
    prev=list(ngrams(prev_sent,n))
    cur=list(ngrams(cur_sent,n))

    common=[x for x in cur if x in prev]
    distance=len(prev)+len(cur)-2*len(common)
    return distance

