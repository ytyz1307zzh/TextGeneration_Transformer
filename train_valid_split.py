keyword_path=r'E:\NLP\datasets\amazon-fine-food-reviews\keywords.txt'
text_path=r'E:\NLP\datasets\amazon-fine-food-reviews\text.txt'
train_src_path=r'E:\NLP\datasets\amazon-fine-food-reviews\amazon_train_source.txt'
train_tgt_path=r'E:\NLP\datasets\amazon-fine-food-reviews\amazon_train_target.txt'
valid_src_path=r'E:\NLP\datasets\amazon-fine-food-reviews\amazon_valid_source.txt'
valid_tgt_path=r'E:\NLP\datasets\amazon-fine-food-reviews\amazon_valid_target.txt'

keyword_file=open(keyword_path,'r',encoding='utf-8')
text_file=open(text_path,'r',encoding='utf-8')
train_src_file=open(train_src_path,'w',encoding='utf-8')
train_tgt_file=open(train_tgt_path,'w',encoding='utf-8')
valid_src_file=open(valid_src_path,'w',encoding='utf-8')
valid_tgt_file=open(valid_tgt_path,'w',encoding='utf-8')

line_cnt=352516
for i in range(line_cnt):
    if i%10000==0:
        print('{} lines finished!'.format(i))
    keyword=keyword_file.readline()
    text=text_file.readline()
    if i < line_cnt*0.8:
        print(keyword,file=train_src_file,end='')
        print(text,file=train_tgt_file,end='')
    else:
        print(keyword,file=valid_src_file,end='')
        print(text,file=valid_tgt_file,end='')

