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


def get_hf_token(cli_token=None, env_file=".env"):
    """Resolve HF token from CLI > environment > .env file."""
    if cli_token:
        return cli_token

    token = os.getenv("HF_TOKEN")
    if token:
        return token

    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "HF_TOKEN":
                    token = v.strip().strip('"').strip("'")
                    if token:
                        os.environ["HF_TOKEN"] = token
                        return token
    return None


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

    # Strip space-padded column names (DAIC-WOZ files can have ' speaker', ' value')
    df.columns = df.columns.str.strip()

    # Filter participant turns — case-insensitive to match any capitalisation
    if 'speaker' in df.columns:
        df = df[df['speaker'].str.strip().str.lower() == 'participant']
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
    p.add_argument("--data_root",  default="C:/Users/ezycloudx-admin/Desktop/thgnn/daicwoz/")
    p.add_argument("--split_csv",  required=True, help="Train or dev CSV with Participant_ID")
    p.add_argument("--model_name", default="mental/mental-bert-base-uncased",
                   help="HuggingFace model. 'mental/mental-bert-base-uncased' recommended for clinical text")
    p.add_argument("--hf_token",   default=None,
                   help="Hugging Face token (optional). If omitted, use HF_TOKEN from env/.env")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len",    type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model_name} on {device}")

    hf_token = get_hf_token(args.hf_token)
    if hf_token:
        print("Using authenticated Hugging Face access.")
    else:
        print("No HF token found. Trying anonymous access.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    model     = AutoModel.from_pretrained(args.model_name, token=hf_token).to(device)

    df = pd.read_csv(args.split_csv)
    df.columns = [c.strip() for c in df.columns]

    for pid in df['Participant_ID']:
        pid = int(pid)
        transcript  = os.path.join(args.data_root, f"{pid}_TRANSCRIPT.csv")
        out_path    = os.path.join(args.data_root, f"{pid}_text_feats.npy")

        # if os.path.exists(out_path):
        #     print(f"  [SKIP] {pid} — already extracted")
        #     continue

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