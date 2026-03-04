#!/usr/bin/env python3
"""
Inference-only script: loads a saved checkpoint and generates predictions
for a given TSV file (dev or test).  Requires no label column.

"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw): return x   # noqa


BATCH_SIZE  = 64
MAX_LENGTH  = 128
DEFAULT_TAU = 0.5


def load_tsv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        first = fh.readline()
        if first.startswith("---"):      # disclaimer present: skip 3 more lines
            for _ in range(3):
                next(fh)
        else:
            fh.seek(0)  # no disclaimer: reset to start of file
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            rows.append({
                "keyword": row[2].strip().lower(),
                "text":    row[4].strip(),
            })
    return rows


class InferDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        enc = self.tokenizer(
            f"Community: {r['keyword']} {r['text']}",
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


@torch.no_grad()
def predict(model_dir: str, input_file: str, output: str) -> None:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(model_dir)

    print(f"Loading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    ).to(device)
    model.eval()

    tau_file = model_dir / "threshold.txt"
    tau      = float(tau_file.read_text().strip()) if tau_file.exists() \
               else DEFAULT_TAU
    print(f"Threshold τ = {tau:.2f}")

    records = load_tsv(input_file)
    print(f"Loaded {len(records):,} examples from {input_file}")

    ds = InferDataset(records, tokenizer, MAX_LENGTH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    probs = []
    for batch in tqdm(dl, desc="Predicting"):
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.squeeze(-1)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    preds = (np.array(probs) >= tau).astype(int).tolist()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        fh.write("\n".join(str(p) for p in preds) + "\n")

    n_pos = sum(preds)
    print(f"Written {len(preds)} predictions to {output} "
          f"({n_pos} positive, {len(preds)-n_pos} negative)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()
    predict(args.model_dir, args.input_file, args.output)
