# TextGeneration_Transformer
text generation from keywords using transformer model

preprocess command:
```bash
python3 preprocess.py -train_src ./transformer/train_source.txt -train_tgt ./transformer/train_target.txt -valid_src ./transformer/valid_source.txt -valid_tgt ./transformer/valid_target.txt -save_data preprocess.pt -max_len 1000
```

train command: 
```bash
python3 train.py -data ./transformer/preprocess.pt -save_model ./saved_model -embs_share_weight -proj_share_weight -emb_path ../glove.6B.300d.txt -log ./log_model -save_mode all -epoch 20 -no_cuda
```

test command:
```bash
python3 translate.py -model saved_model_accu_39.338.chkpt -src ./transformer/valid_source.txt -vocab ./transformer/preprocess.pt -output ./valid_pred.txt -no_cuda
```
