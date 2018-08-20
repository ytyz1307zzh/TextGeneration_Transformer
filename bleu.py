from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction

pred_path='./valid_pred.txt'
tgt_path='./transformer/valid_target.txt'
predictions=open(pred_path,'r',encoding='utf-8')
targets=open(tgt_path,'r',encoding='utf-8')
references=[]
hypotheses=[]

for tgt in targets:
    references.append([tgt.strip().split()])

for pred in predictions:
    hypotheses.append(pred.strip().split())

if len(references)!=len(hypotheses):
    print('The number of reference sentences doesn\'t match the predicted sentences.')
    quit()

score=corpus_bleu(references,hypotheses,smoothing_function=SmoothingFunction().method1)
print('BLEU score is '+str(score*100))
