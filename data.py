import re
import pandas as pd
import numpy as np

def erase_noise(string):
    string=re.sub('<.*>',' ',string)
    string=re.sub(r'\"',' \" ',string)
    string=re.sub(r',',' ,',string)
    string=re.sub(r'\.',' .',string)
    string=re.sub(r':',' :',string)
    string=re.sub(r';',' ;',string)
    string=re.sub(r'\?',' ?',string)
    string=re.sub(r'!',' !',string)
    string=re.sub(r'\(',' ( ',string)
    string=re.sub(r'\)',' ) ',string)
    string=re.sub(r'&','and',string)
    string=re.sub(r'\'s',' \'s',string)
    string=re.sub(r'\'ve',' \'ve',string)
    string=re.sub(r'\'d',' \'d',string)
    string=re.sub(r'\'ll',' \'ll',string)
    string=re.sub(r'\'re',' \'re',string)
    string=re.sub(r'\'m',' \'m',string)
    string=re.sub(r'n\'t',' n\'t',string)
    string=re.sub(r'\. \. \.','...',string)
    string=string.lower()
    return string


data_path=r'E:\NLP\datasets\amazon-fine-food-reviews\Reviews.csv'
keyword_path=r'E:\NLP\datasets\amazon-fine-food-reviews\keywords.txt'
text_path=r'E:\NLP\datasets\amazon-fine-food-reviews\text.txt'
keyword_file=open(keyword_path,'w',encoding='utf-8')
text_file=open(text_path,'w',encoding='utf-8')

line_cnt=0
omit_cnt=0
data=pd.read_csv(data_path,dtype=str)
titles=data['Summary']
texts=data['Text']
lines=zip(titles,texts)

for title,text in lines:
    title=str(title)
    text=str(text)
    line_cnt+=1
    if line_cnt%10000==0:
        print('{} lines finished!'.format(line_cnt))
    if line_cnt==1:
        continue
    if len(title.split())<2: # omit samples with too few keywords
        omit_cnt+=1
        continue
    title=erase_noise(title)
    text=erase_noise(text)
    if len(text.split())>200: # omit samples with too many words in the text
        omit_cnt+=1
        continue
    if len(text.split())<20: # omit samples with too few words in the text
        omit_cnt+=1
        continue
    print(title,file=keyword_file)
    print(text,file=text_file)


print('Total lines (before omission): {}'.format(line_cnt))
print('{} lines omitted'.format(omit_cnt))
print('{} lines remaining'.format(line_cnt-omit_cnt))

