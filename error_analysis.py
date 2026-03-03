"""
The local evaluation for the PCL detection task.

Uses the identical 90/10 split as BestModel/baseline_cpu.py (SEED=42)
so results are directly comparable.
"""

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline

# ── Constants (must match baseline_cpu.py) ────────────────────────────────────
SEED       = 42
DEV_FRAC   = 0.10
TSV_PATH   = Path("Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv")
FIG_DIR    = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

TAU_GRID   = np.arange(0.10, 0.91, 0.01)

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "figure.dpi":        160,
})

C_NEG   = "#4878CF"
C_POS   = "#D65F5F"
C_FNFP  = ["#ef6351", "#4878CF", "#43aa8b", "#f9c74f"]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_tsv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for _ in range(4):
            next(fh)
        for row in csv.reader(fh, delimiter="\t"):
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
                "raw_label": raw,
            })
    return rows


def build_input(kw: str, text: str) -> str:
    return f"Community: {kw} {text}"


def tune_tau(probs, gold):
    best_tau, best_f1 = 0.5, 0.0
    for tau in TAU_GRID:
        f1 = f1_score(gold, (probs >= tau).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return round(best_tau, 2), round(best_f1, 4)


def latex_safe(text: str, max_words: int = 35) -> str:
    """Truncate and escape text for LaTeX inclusion."""
    words = text.split()
    truncated = " ".join(words[:max_words])
    if len(words) > max_words:
        truncated += r" [\ldots]"
    # escape LaTeX special chars
    for ch, rep in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                    ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                    ("}", r"\}"), ("~", r"\textasciitilde{}"),
                    ("^", r"\textasciicircum{}"), ("\\", r"\textbackslash{}")]:
        truncated = truncated.replace(ch, rep)
    return truncated


# ── Build and evaluate a model configuration ──────────────────────────────────
def evaluate_config(
    train_texts, train_labels,
    dev_texts,   dev_labels,
    use_prefix:  bool  = True,
    use_balance: bool  = True,
    tau:         float = None,
) -> dict:
    """Train a TF-IDF+LR variant and return metrics dict.
    Caller is responsible for building train_texts / dev_texts
    (with or without keyword prefix) before calling this function.
    use_prefix and use_balance are accepted for documentation only."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2,
            max_features=80_000, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced" if use_balance else None,
            C=1.0, max_iter=1000,
            random_state=SEED, solver="lbfgs",
        )),
    ])
    pipe.fit(train_texts, train_labels)
    probs = pipe.predict_proba(dev_texts)[:, 1]

    if tau is None:
        tau, _ = tune_tau(probs, dev_labels)

    preds = (probs >= tau).astype(int)
    return {
        "pipe":      pipe,
        "probs":     probs,
        "preds":     preds,
        "tau":       tau,
        "f1":        round(f1_score(dev_labels, preds, zero_division=0),   4),
        "precision": round(precision_score(dev_labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(dev_labels, preds, zero_division=0),    4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # ── Load + split (identical to baseline_cpu.py) ────────────────────────
    all_records = load_tsv(TSV_PATH)
    random.shuffle(all_records)
    cut   = int((1 - DEV_FRAC) * len(all_records))
    train = all_records[:cut]
    dev   = all_records[cut:]

    # Full model (C2+C3)
    tr_x = [build_input(r["keyword"], r["text"]) for r in train]
    tr_y = [r["label"] for r in train]
    dv_x = [build_input(r["keyword"], r["text"]) for r in dev]
    dv_y = [r["label"] for r in dev]

    full = evaluate_config(tr_x, tr_y, dv_x, dv_y,
                           use_prefix=True, use_balance=True)

    preds  = full["preds"]
    probs  = full["probs"]
    tau    = full["tau"]

    # ── Classification report ──────────────────────────────────────────────
    SEP = "=" * 65
    print(SEP)
    print("FULL MODEL — Classification Report")
    print(SEP)
    print(classification_report(dv_y, preds,
                                target_names=["No-PCL", "PCL"],
                                zero_division=0))
    cm = confusion_matrix(dv_y, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Threshold τ = {tau:.2f}")
    print()

    # ── Error catalogue ────────────────────────────────────────────────────
    fn_recs = [r for r, g, p in zip(dev, dv_y, preds) if g == 1 and p == 0]
    fp_recs = [r for r, g, p in zip(dev, dv_y, preds) if g == 0 and p == 1]
    tp_recs = [r for r, g, p in zip(dev, dv_y, preds) if g == 1 and p == 1]

    print(SEP)
    print("ERROR CATALOGUE")
    print(SEP)
    print(f"  False Negatives (missed PCL) : {len(fn_recs)}")
    print(f"  False Positives (false alarm) : {len(fp_recs)}")
    print(f"  True Positives               : {len(tp_recs)}")
    print()

    # Per-keyword FN/FP
    kw_fn = Counter(r["keyword"] for r in fn_recs)
    kw_fp = Counter(r["keyword"] for r in fp_recs)
    kw_tp = Counter(r["keyword"] for r in tp_recs)
    kw_all_dev = Counter(r["keyword"] for r in dev if r["label"] == 1)

    print(SEP)
    print("PER-KEYWORD ERROR BREAKDOWN  (dev positive examples)")
    print(SEP)
    print(f"  {'Keyword':<16}  {'Total PCL':>9}  {'TP':>4}  {'FN':>4}  "
          f"{'FP':>4}  {'Recall':>7}")
    print("  " + "-" * 54)
    for kw in sorted(kw_all_dev, key=lambda k: -kw_all_dev[k]):
        tot  = kw_all_dev[kw]
        tp_k = kw_tp.get(kw, 0)
        fn_k = kw_fn.get(kw, 0)
        fp_k = kw_fp.get(kw, 0)
        rec  = tp_k / max(tot, 1)
        print(f"  {kw:<16}  {tot:>9}  {tp_k:>4}  {fn_k:>4}  "
              f"{fp_k:>4}  {rec:>6.1%}")
    print()

    # ── Select illustrative examples ───────────────────────────────────────
    # Sort FNs by model probability (hardest: lowest prob → most confidently wrong)
    fn_sorted = sorted(
        zip(fn_recs, [probs[i] for i, (g, p) in
                      enumerate(zip(dv_y, preds)) if g == 1 and p == 0]),
        key=lambda x: x[1]
    )
    fp_sorted = sorted(
        zip(fp_recs, [probs[i] for i, (g, p) in
                      enumerate(zip(dv_y, preds)) if g == 0 and p == 1]),
        key=lambda x: -x[1]
    )

    print(SEP)
    print("SELECTED FALSE NEGATIVES  (most confidently wrong first)")
    print(SEP)
    for i, (r, prob) in enumerate(fn_sorted[:5], 1):
        wc = len(r["text"].split())
        print(f"  FN-{i}  keyword={r['keyword']:<16}  "
              f"raw_label={r['raw_label']}  prob={prob:.3f}  wc={wc}")
        print(f"  TEXT: {r['text'][:200]}")
        print()

    print(SEP)
    print("SELECTED FALSE POSITIVES  (most confidently wrong first)")
    print(SEP)
    for i, (r, prob) in enumerate(fp_sorted[:4], 1):
        wc = len(r["text"].split())
        print(f"  FP-{i}  keyword={r['keyword']:<16}  "
              f"raw_label={r['raw_label']}  prob={prob:.3f}  wc={wc}")
        print(f"  TEXT: {r['text'][:200]}")
        print()

    # ── Ablation study ─────────────────────────────────────────────────────
    # Use the threshold tuned by the full model for fair comparison
    ablation_tau = tau

    # A: no prefix, no balance
    tr_x_nopfx = [r["text"] for r in train]
    dv_x_nopfx = [r["text"] for r in dev]
    cfg_a = evaluate_config(tr_x_nopfx, tr_y, dv_x_nopfx, dv_y,
                            use_prefix=False, use_balance=False, tau=ablation_tau)

    # B: no prefix, with balance
    cfg_b = evaluate_config(tr_x_nopfx, tr_y, dv_x_nopfx, dv_y,
                            use_prefix=False, use_balance=True, tau=ablation_tau)

    # C: prefix, no balance
    cfg_c = evaluate_config(tr_x, tr_y, dv_x, dv_y,
                            use_prefix=True, use_balance=False, tau=ablation_tau)

    # D: prefix + balance (full — re-use fixed tau for fair comparison)
    cfg_d = evaluate_config(tr_x, tr_y, dv_x, dv_y,
                            use_prefix=True, use_balance=True, tau=ablation_tau)

    ablation = [
        ("A: No prefix, no balance",   cfg_a),
        ("B: No prefix, + balance",    cfg_b),
        ("C: + Prefix,  no balance",   cfg_c),
        ("D: + Prefix,  + balance",    cfg_d),
    ]

    print(SEP)
    print(f"ABLATION STUDY  (fixed τ={ablation_tau:.2f})")
    print(SEP)
    print(f"  {'Config':<28}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print("  " + "-" * 52)
    for name, cfg in ablation:
        print(f"  {name:<28}  {cfg['precision']:>6.4f}  "
              f"{cfg['recall']:>6.4f}  {cfg['f1']:>6.4f}")
    print()

    # ── LaTeX example table snippet ────────────────────────────────────────
    print(SEP)
    print("LATEX: False Negative examples (copy into report)")
    print(SEP)
    for i, (r, prob) in enumerate(fn_sorted[:4], 1):
        wc  = len(r["text"].split())
        txt = latex_safe(r["text"], max_words=30)
        print(f"  FN-{i} & {r['keyword']} & {r['raw_label']} & {prob:.2f} & "
              f"\\textit{{{txt}}} \\\\")
    print()
    print(SEP)
    print("LATEX: False Positive examples")
    print(SEP)
    for i, (r, prob) in enumerate(fp_sorted[:3], 1):
        wc  = len(r["text"].split())
        txt = latex_safe(r["text"], max_words=30)
        print(f"  FP-{i} & {r['keyword']} & {r['raw_label']} & {prob:.2f} & "
              f"\\textit{{{txt}}} \\\\")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 3 – Confusion Matrix
    # ═══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(cm, interpolation="nearest",
                   cmap=plt.cm.Blues, vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    tick_labels = ["No-PCL\n(Actual 0)", "PCL\n(Actual 1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0\n(No-PCL)", "Pred 1\n(PCL)"], fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    ax.set_title(f"Confusion Matrix (dev split, τ={tau:.2f})", fontsize=10)

    total_dev = len(dv_y)
    labels_cm = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = 100 * val / total_dev
            colour = "white" if val > cm.max() * 0.6 else "black"
            ax.text(j, i, f"{labels_cm[i][j]}\n{val}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=colour)

    fig.tight_layout()
    out3 = FIG_DIR / "fig3_confusion_matrix.pdf"
    fig.savefig(out3, bbox_inches="tight", format="pdf")
    print(f"Saved → {out3}")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 4 – Error Breakdown + Ablation
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): per-keyword FN/FP
    ax = axes[0]
    kws = sorted(kw_all_dev, key=lambda k: -kw_all_dev[k])
    fn_vals = [kw_fn.get(k, 0) for k in kws]
    fp_vals = [kw_fp.get(k, 0) for k in kws]
    tp_vals = [kw_tp.get(k, 0) for k in kws]
    x = np.arange(len(kws))
    w = 0.28
    ax.bar(x - w, tp_vals, width=w, color="#43aa8b", label="TP (correct PCL)",   zorder=3)
    ax.bar(x,     fn_vals, width=w, color=C_POS,     label="FN (missed PCL)",    zorder=3)
    ax.bar(x + w, fp_vals, width=w, color=C_NEG,     label="FP (false alarm)",   zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(kws, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("(a) Per-Keyword Error Breakdown", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Panel (b): ablation F1 / P / R
    ax = axes[1]
    abl_names  = ["A\nNo prefix\nNo balance",
                  "B\nNo prefix\n+Balance",
                  "C\n+Prefix\nNo balance",
                  "D\n+Prefix\n+Balance"]
    abl_f1  = [cfg["f1"]        for _, cfg in ablation]
    abl_p   = [cfg["precision"] for _, cfg in ablation]
    abl_r   = [cfg["recall"]    for _, cfg in ablation]
    x2 = np.arange(len(abl_names))
    w2 = 0.24
    b1 = ax.bar(x2 - w2, abl_p,  width=w2, color="#4878CF", label="Precision", zorder=3)
    b2 = ax.bar(x2,       abl_r,  width=w2, color="#e07b54", label="Recall",    zorder=3)
    b3 = ax.bar(x2 + w2, abl_f1, width=w2, color="#43aa8b", label="F1",        zorder=3)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x2)
    ax.set_xticklabels(abl_names, fontsize=8)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score")
    ax.set_title(f"(b) Ablation Study (τ={ablation_tau:.2f} fixed)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=1.8)
    out4 = FIG_DIR / "fig4_error_breakdown.pdf"
    fig.savefig(out4, bbox_inches="tight", format="pdf")
    print(f"Saved → {out4}")
    plt.close(fig)

    print("\nPhase 5 error analysis complete.")


if __name__ == "__main__":
    main()
