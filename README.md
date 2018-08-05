# TextGeneration_Transformer
text generation from keywords using transformer model

train command: 
```bash
python3 train.py -data ./transformer/preprocess.pt -save_model ./saved_model -embs_share_weight -proj_share_weight -emb_path ../glove.6B.300d.txt -log ./log_model -save_mode all -no_cuda
```
