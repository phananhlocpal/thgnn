# Bước 1: Extract BERT cả 3 context modes
python extract_bert_daicwoz.py --split_csv daicwoz/train_split_Depression_AVEC2017.csv 
python extract_bert_daicwoz.py --split_csv daicwoz/dev_split_Depression_AVEC2017.csv 
python extract_bert_daicwoz.py --split_csv daicwoz/test_split_Depression_AVEC2017.csv 

python extract_wav2vec_daicwoz.py --split_csv daicwoz/train_split_Depression_AVEC2017.csv --fp16 --batch_size 32
python extract_wav2vec_daicwoz.py --split_csv daicwoz/dev_split_Depression_AVEC2017.csv --fp16 --batch_size 32
python extract_wav2vec_daicwoz.py --split_csv daicwoz/test_split_Depression_AVEC2017.csv --fp16 --batch_size 32
# Bước 2: Train multi-seed với mode "none" (anti-shortcut)
python train.py --dataset daicwoz --context_mode none --multi_seed 

# Bước 3: Train multi-seed với mode "truncated" để so sánh
python train.py --dataset daicwoz --context_mode truncated --multi_seed

# Nếu F1(none) ≈ F1(truncated) → model học genuine signals → safe to claim