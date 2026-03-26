"""
Pre-extract BERT embeddings from DAIC-WOZ transcripts.
Run this ONCE before training.

Data structure:
  data/daicwoz/
    {ID}_TRANSCRIPT.csv
    (after running this script)
    {ID}_text_feats.npy         ← Generated here

Usage:
    python extract_bert.py --split_csv data/daicwoz/train_split_Depression_AVEC2017.csv
    python extract_bert.py --split_csv data/daicwoz/dev_split_Depression_AVEC2017.csv

Saves: data/daicwoz/{ID}_text_feats.npy — shape (N_turns, 768)

Requires: pip install transformers torch
"""

import os
import argparse
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel


def mean_pool(token_embeddings, attention_mask):
    """Mean pooling over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def extract_bert_features(
    transcript_path,
    tokenizer,
    model,
    device,
    model_name="bert-base-uncased",
    max_len=128,
    batch_size=32,
):
    """
    Extract per-turn BERT embeddings from a DAIC-WOZ transcript CSV.
    Only Participant turns are used.
    """
    try:
        df = pd.read_csv(transcript_path, sep='\t', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(transcript_path, on_bad_lines='skip')

    # Filter participant turns
    if 'speaker' in df.columns:
        df = df[df['speaker'].str.lower() == 'participant']
    if 'value' not in df.columns:
        df['value'] = ''

    turns = df['value'].fillna('').astype(str).tolist()
    if not turns:
        turns = ['[empty]']

    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(turns), batch_size):
            batch_texts = turns[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt',
            ).to(device)

            out = model(**enc)
            # Use CLS token OR mean pool — mean pool is more robust
            emb = mean_pool(out.last_hidden_state, enc['attention_mask'])
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="data/daicwoz/")
    p.add_argument("--split_csv",  required=True, help="Train or dev CSV with Participant_ID")
    p.add_argument("--model_name", default="bert-base-uncased",
                   help="HuggingFace model. 'mental/mental-bert-base-uncased' recommended for clinical text")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len",    type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModel.from_pretrained(args.model_name).to(device)

    df = pd.read_csv(args.split_csv)
    df.columns = [c.strip() for c in df.columns]

    for pid in df['Participant_ID']:
        pid = int(pid)
        transcript  = os.path.join(args.data_root, f"{pid}_TRANSCRIPT.csv")
        out_path    = os.path.join(args.data_root, f"{pid}_text_feats.npy")

        if os.path.exists(out_path):
            print(f"  [SKIP] {pid} — already extracted")
            continue

        if not os.path.exists(transcript):
            print(f"  [MISS] {pid} — transcript not found: {transcript}")
            continue

        print(f"  Processing participant {pid} ...")
        feats = extract_bert_features(
            transcript, tokenizer, model, device,
            model_name=args.model_name,
            max_len=args.max_len,
            batch_size=args.batch_size,
        )
        np.save(out_path, feats)
        print(f"    Saved: {out_path}  shape={feats.shape}")

    print("\nBERT feature extraction complete.")
    print("TIP: Consider using 'mental/mental-bert-base-uncased' for better")
    print("     clinical language understanding in depression detection tasks.")


if __name__ == "__main__":
    main()