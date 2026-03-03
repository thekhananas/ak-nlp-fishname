"""
This script provides a verification that the data-loading
and prediction pipeline is correct.  It is NOT the BestModel submission —
the actual submission uses DeBERTa-v3-base (train.py).
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline

SEED           = 42
DEV_FRAC       = 0.10
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.01)


# ── Data loading ──
def load_tsv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for _ in range(4):
            next(fh)
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            try:
                raw = int(row[5].strip())
            except ValueError:
                continue
            rows.append({
                "keyword": row[2].strip().lower(),
                "text":    row[4].strip(),
                "label":   1 if raw >= 2 else 0,
            })
    return rows


def build_input(keyword: str, text: str) -> str:
    return f"Community: {keyword} {text}"


def tune_threshold(
    probs: np.ndarray, gold: list[int]
) -> tuple[float, float]:
    best_tau, best_f1 = 0.5, 0.0
    for tau in THRESHOLD_GRID:
        preds = (probs >= tau).astype(int)
        f1    = f1_score(gold, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return round(best_tau, 2), round(best_f1, 4)


# ── Main ──
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--dev_out",    default="dev.txt")
    parser.add_argument("--test_out",   default="test.txt")
    parser.add_argument("--test_lines", type=int, default=3832,
                        help="Number of lines in test.txt placeholder.")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    # ── Load + split ────
    print("Loading data...")
    all_records = load_tsv(args.train_file)
    random.shuffle(all_records)
    cut = int((1 - DEV_FRAC) * len(all_records))
    train_records = all_records[:cut]
    dev_records   = all_records[cut:]

    train_texts  = [build_input(r["keyword"], r["text"]) for r in train_records]
    train_labels = [r["label"] for r in train_records]
    dev_texts    = [build_input(r["keyword"], r["text"]) for r in dev_records]
    dev_labels   = [r["label"] for r in dev_records]

    n_pos_train = sum(train_labels)
    n_pos_dev   = sum(dev_labels)
    print(f"  Train : {len(train_records):,}  (PCL={n_pos_train})")
    print(f"  Dev   : {len(dev_records):,}  (PCL={n_pos_dev})")

    # ── TF-IDF + Logistic Regression  (C2: class_weight='balanced') ───────
    print("\nFitting TF-IDF + Logistic Regression ...")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",   # C2: counters imbalance
            C=1.0,
            max_iter=1000,
            random_state=SEED,
            solver="lbfgs",
        )),
    ])
    pipe.fit(train_texts, train_labels)

    # ── Threshold tuning (C4) ────    
    dev_probs = pipe.predict_proba(dev_texts)[:, 1]
    tau, best_f1 = tune_threshold(dev_probs, dev_labels)
    dev_preds = (dev_probs >= tau).astype(int).tolist()

    print(f"\nDev results (τ={tau:.2f}):")
    print(f"  F1 (positive class) = {best_f1:.4f}")
    print(classification_report(
        dev_labels, dev_preds,
        target_names=["No-PCL", "PCL"],
        zero_division=0,
    ))

    # ── Write dev.txt ────
    Path(args.dev_out).write_text(
        "\n".join(str(p) for p in dev_preds) + "\n"
    )
    print(f"Written: {args.dev_out}  ({len(dev_preds)} lines, "
          f"{sum(dev_preds)} predicted PCL)")

    
    # Replace with actual DeBERTa predictions by running:
    #   python BestModel/train.py --train_file ... --test_file ... --test_out test.txt
    placeholder = ["0"] * args.test_lines
    Path(args.test_out).write_text("\n".join(placeholder) + "\n")
    print(f"Written: {args.test_out}  ({args.test_lines} placeholder zeros)")
    print("\n[!] test.txt is a format placeholder — replace with "
          "DeBERTa predictions after GPU training.")


if __name__ == "__main__":
    main()
