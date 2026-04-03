# Bước 1: Extract cả 3 context modes
python extract_bert_v3_fixed.py --split_csv train.csv --all_modes

# Bước 2: Train multi-seed với mode "none" (anti-shortcut)
python train.py --dataset daicwoz --context_mode none --multi_seed

# Bước 3: Train multi-seed với mode "truncated" để so sánh
python train.py --dataset daicwoz --context_mode truncated --multi_seed

# Nếu F1(none) ≈ F1(truncated) → model học genuine signals → safe to claim