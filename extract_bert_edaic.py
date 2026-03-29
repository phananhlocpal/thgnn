"""
Pre-extract BERT embeddings from E-DAIC transcripts.

E-DAIC transcript format:
  {ID}_Transcript.csv — comma-separated
  Columns: Start_Time, End_Time, Text, Confidence
  NO speaker column (all text is participant)

Usage:
    python extract_bert.py --split_csv data/edaic/train_split.csv --data_root data/edaic/
    python extract_bert.py --split_csv data/edaic/dev_split.csv --data_root data/edaic/

Saves: data/edaic/{ID}_text_feats.npy — shape (N_turns, 768)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def get_hf_token(cli_token=None):
    if cli_token:
        return cli_token
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def extract_bert_features(transcript_path, tokenizer, model, device,
                          max_len=128, batch_size=32):
    """
    Extract per-turn BERT embeddings from E-DAIC transcript.
    E-DAIC format: comma-sep, columns: Start_Time, End_Time, Text, Confidence
    All rows are participant text (no speaker filtering needed).
    """
    try:
        df = pd.read_csv(transcript_path, on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(transcript_path)

    # Strip space-padded column names
    df.columns = df.columns.str.strip()

    # Find text column
    text_col = None
    for c in ['Text', 'text', 'value', 'Value']:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]

    # IMPORTANT: preserve ALL rows (including empty text) to stay aligned
    # with the transcript row count used in dataset.py _load_transcript.
    # Empty turns → '[EMPTY]' placeholder so BERT still produces a valid embedding.
    turns = df[text_col].fillna('').astype(str).tolist()
    turns = [t.strip() if t.strip() else '[EMPTY]' for t in turns]
    if not turns:
        turns = ['[EMPTY]']

    all_emb = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(turns), batch_size):
            batch = turns[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt').to(device)
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc['attention_mask'])
            all_emb.append(emb.cpu().numpy())

    return np.vstack(all_emb).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="C:/Users/ezycloudx-admin/Desktop/thgnn/edaic_final/")
    p.add_argument("--split_csv", required=True)
    p.add_argument("--model_name", default="mental/mental-bert-base-uncased")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model_name} on {device}")

    hf_token = get_hf_token(args.hf_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    model = AutoModel.from_pretrained(args.model_name, token=hf_token).to(device)

    df = pd.read_csv(args.split_csv)
    df.columns = [c.strip() for c in df.columns]

    # Auto-detect ID column
    id_col = None
    for c in ['Participant_ID', 'participant_id', 'ID', 'id']:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    for pid in df[id_col]:
        pid = int(pid)
        # E-DAIC naming convention
        transcript = os.path.join(args.data_root, f"{pid}_Transcript.csv")
        out_path = os.path.join(args.data_root, f"{pid}_text_feats.npy")

        if not os.path.exists(transcript):
            print(f"  [MISS] {pid} — transcript not found")
            continue

        print(f"  Processing {pid} ...")
        feats = extract_bert_features(
            transcript, tokenizer, model, device,
            max_len=args.max_len, batch_size=args.batch_size)
        np.save(out_path, feats)
        print(f"    Saved: {out_path} shape={feats.shape}")

    print("\nBERT feature extraction complete.")


if __name__ == "__main__":
    main()