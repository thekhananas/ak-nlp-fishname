#!/usr/bin/env python3
"""
Fine-tunes microsoft/deberta-v3-base for binary PCL detection.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw):   # noqa: E302
        return x

# ── Constants ─────────────────────────────────────────────────────────────────
SEED           = 42
MODEL_NAME     = "microsoft/deberta-v3-base"
MAX_LENGTH     = 128          # covers ~95 % of corpus at default tokenisation
BATCH_SIZE     = 32
LEARNING_RATE  = 2e-5
EPOCHS         = 10
WARMUP_RATIO   = 0.06         # 6 % of total steps
PATIENCE       = 3            # early-stopping patience on dev F1
POS_WEIGHT     = 9.54         # empirical neg:pos ratio from EDA
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.01)   # C4: threshold sweep


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_tsv(path: str, has_labels: bool = True) -> list[dict]:
    """
    Load a DPM! TSV file, skipping the 4-line disclaimer header.

    Columns: par_id | art_id | keyword | country_code | text | [label]

    Returns a list of dicts with keys:
        par_id, keyword, text, label  (-1 when labels unavailable)
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for _ in range(4):          # skip disclaimer
            next(fh)
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            entry = {
                "par_id":  row[0].strip(),
                "keyword": row[2].strip().lower(),
                "text":    row[4].strip(),
                "label":   -1,
            }
            if has_labels and len(row) >= 6:
                try:
                    raw = int(row[5].strip())
                    entry["label"] = 1 if raw >= 2 else 0   # {2,3,4} → PCL
                except ValueError:
                    continue
            rows.append(entry)
    return rows


# ── Input construction (C3) ───────────────────────────────────────────────────
def build_input(keyword: str, text: str) -> str:
    """Prepend vulnerability keyword as a community-type conditioning signal."""
    return f"Community: {keyword} {text}"


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class PCLDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r   = self.records[idx]
        enc = self.tokenizer(
            build_input(r["keyword"], r["text"]),
            max_length  = self.max_length,
            padding     = "max_length",
            truncation  = True,
            return_tensors = "pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if r["label"] != -1:
            item["labels"] = torch.tensor(r["label"], dtype=torch.float)
        return item


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(
    model, loader, optimizer, scheduler, loss_fn, device, scaler=None
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}

        if scaler is not None:                          # mixed precision
            with torch.cuda.amp.autocast():
                logits = model(**batch).logits.squeeze(-1)
                loss   = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(**batch).logits.squeeze(-1)
            loss   = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def get_probs(model, loader, device) -> np.ndarray:
    """Run inference and return sigmoid probabilities."""
    model.eval()
    probs = []
    for batch in tqdm(loader, desc="  infer", leave=False):
        batch.pop("labels", None)
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.squeeze(-1)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(probs)


def tune_threshold(
    probs: np.ndarray, gold_labels: list[int]
) -> tuple[float, float]:
    """Sweep THRESHOLD_GRID; return (best_τ, best_F1_positive_class)."""
    best_tau, best_f1 = 0.5, 0.0
    for tau in THRESHOLD_GRID:
        preds = (probs >= tau).astype(int)
        f1    = f1_score(gold_labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return round(best_tau, 2), round(best_f1, 4)


def write_predictions(preds: list[int], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(str(p) for p in preds) + "\n")
    print(f"  Written: {path}  ({len(preds)} lines)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeBERTa-v3-base fine-tuning for PCL binary detection."
    )
    parser.add_argument("--train_file",  required=True,
                        help="Path to dontpatronizeme_pcl.tsv (training set).")
    parser.add_argument("--dev_file",    default=None,
                        help="Path to official dev TSV (with labels).")
    parser.add_argument("--test_file",   default=None,
                        help="Path to official test TSV (labels optional).")
    parser.add_argument("--output_dir",  default="BestModel/checkpoints",
                        help="Directory for saving model checkpoints.")
    parser.add_argument("--dev_out",     default="dev.txt",
                        help="Output path for dev-set predictions.")
    parser.add_argument("--test_out",    default="test.txt",
                        help="Output path for test-set predictions.")
    parser.add_argument("--split_dev",   action="store_true",
                        help="Hold out 10 %% of training data as dev set.")
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LEARNING_RATE)
    parser.add_argument("--max_length",  type=int,   default=MAX_LENGTH)
    parser.add_argument("--pos_weight",  type=float, default=POS_WEIGHT,
                        help="Positive-class weight for BCEWithLogitsLoss.")
    parser.add_argument("--seed",        type=int,   default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Model       : {MODEL_NAME}")
    print(f"Max length  : {args.max_length} tokens")
    print(f"Pos weight  : {args.pos_weight}")
    print(f"Batch size  : {args.batch_size}")
    print(f"LR          : {args.lr}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    train_records = load_tsv(args.train_file, has_labels=True)

    if args.dev_file:
        dev_records = load_tsv(args.dev_file, has_labels=True)
    elif args.split_dev:
        random.shuffle(train_records)
        cut = int(0.90 * len(train_records))
        train_records, dev_records = train_records[:cut], train_records[cut:]
    else:
        dev_records = None

    test_records = load_tsv(args.test_file, has_labels=False) \
        if args.test_file else None

    n_dev  = len(dev_records)  if dev_records  else 0
    n_test = len(test_records) if test_records else 0
    print(f"  Train : {len(train_records):>6,}")
    print(f"  Dev   : {n_dev:>6,}  {'(internal split)' if args.split_dev else ''}")
    print(f"  Test  : {n_test:>6,}\n")

    # ── Tokeniser + DataLoaders ────────────────────────────────────────────
    print(f"Loading tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = PCLDataset(train_records, tokenizer, args.max_length)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"),
    )

    dev_dl = test_dl = None
    if dev_records:
        dev_ds = PCLDataset(dev_records, tokenizer, args.max_length)
        dev_dl = DataLoader(
            dev_ds, batch_size=args.batch_size * 2,
            shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"),
        )
    if test_records:
        test_ds = PCLDataset(test_records, tokenizer, args.max_length)
        test_dl = DataLoader(
            test_ds, batch_size=args.batch_size * 2,
            shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"),
        )

    # ── Model, optimiser, loss ─────────────────────────────────────────────
    print(f"Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1          # single logit → sigmoid
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
    )
    total_steps  = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # C2: class-weighted loss
    pos_weight = torch.tensor([args.pos_weight], device=device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Automatic mixed precision (GPU only)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    
    best_f1, best_tau   = 0.0, 0.5
    patience_counter    = 0
    best_ckpt           = output_dir / "best_model"

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(early-stop patience = {PATIENCE})...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_dl, optimizer, scheduler, loss_fn, device, scaler
        )

        if dev_dl is not None:
            gold   = [r["label"] for r in dev_records]
            probs  = get_probs(model, dev_dl, device)
            tau, f1 = tune_threshold(probs, gold)
            print(f"Epoch {epoch:02d}  loss={train_loss:.4f}  "
                  f"dev_F1={f1:.4f}  τ={tau:.2f}")

            if f1 > best_f1:
                best_f1, best_tau = f1, tau
                patience_counter  = 0
                model.save_pretrained(best_ckpt)
                tokenizer.save_pretrained(best_ckpt)
                (best_ckpt / "threshold.txt").write_text(str(best_tau))
                print(f"  ✓ Saved best  (F1={best_f1:.4f}  τ={best_tau:.2f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break
        else:
            print(f"Epoch {epoch:02d}  loss={train_loss:.4f}")
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)

    
    if best_ckpt.exists():
        print(f"\nReloading best checkpoint → {best_ckpt}")
        model = AutoModelForSequenceClassification.from_pretrained(
            best_ckpt
        ).to(device)
        tau_file = best_ckpt / "threshold.txt"
        if tau_file.exists():
            best_tau = float(tau_file.read_text().strip())
        print(f"  Threshold τ = {best_tau:.2f}")

    
    if dev_dl is not None:
        print(f"\nGenerating {args.dev_out} ...")
        gold  = [r["label"] for r in dev_records]
        probs = get_probs(model, dev_dl, device)
        preds = (probs >= best_tau).astype(int).tolist()
        final_f1 = f1_score(gold, preds, pos_label=1, zero_division=0)
        print(f"  Final dev F1 (τ={best_tau:.2f}) = {final_f1:.4f}")
        print(classification_report(
            gold, preds, target_names=["No-PCL", "PCL"], zero_division=0
        ))
        write_predictions(preds, args.dev_out)

    
    if test_dl is not None:
        print(f"\nGenerating {args.test_out} ...")
        probs = get_probs(model, test_dl, device)
        preds = (probs >= best_tau).astype(int).tolist()
        write_predictions(preds, args.test_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
