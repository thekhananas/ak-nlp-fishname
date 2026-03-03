"""
EDA for the DPM! PCL detection task.
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
TSV     = BASE / "Dont_Patronize_Me_Trainingset" / "dontpatronizeme_pcl.tsv"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    8.5,
    "figure.dpi":         160,
})

# ── Colours ───────────────────────────────────────────────────────────────────
C_NEG     = "#4878CF"     # blue  – No-PCL
C_POS     = "#D65F5F"     # red   – PCL
PALETTE_5 = ["#3a86ff", "#8ecae6", "#fcbf49", "#ef6351", "#d62828"]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for _ in range(4):          # skip 4-line disclaimer header
            next(f)
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            try:
                rows.append({
                    "par_id":  row[0],
                    "art_id":  row[1],
                    "keyword": row[2].strip().lower(),
                    "country": row[3].strip().lower(),
                    "text":    row[4],
                    "label":   int(row[5]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def word_count(text: str) -> int:
    return len(text.split())


# ── Stats helpers ─────────────────────────────────────────────────────────────
def pct(n, total): return 100 * n / total


# ═════════════════════════════════════════════════════════════════════════════
def main():
    data  = load_data(TSV)
    total = len(data)
    print(f"Loaded {total:,} paragraphs.\n")

    # Binary mapping  {0,1} → 0 (No-PCL)   {2,3,4} → 1 (PCL)
    for r in data:
        r["binary"] = 1 if r["label"] >= 2 else 0
        r["wc"]     = word_count(r["text"])

    labels   = [r["label"]  for r in data]
    binary   = [r["binary"] for r in data]
    wc_all   = np.array([r["wc"] for r in data])
    wc_neg   = np.array([r["wc"] for r in data if r["binary"] == 0])
    wc_pos   = np.array([r["wc"] for r in data if r["binary"] == 1])
    keywords = [r["keyword"] for r in data]

    counts5  = Counter(labels)
    counts2  = Counter(binary)
    n_neg, n_pos = counts2[0], counts2[1]

    # ── Console tables ────────────────────────────────────────────────────────
    SEP = "=" * 60

    print(SEP)
    print("TABLE 1: Five-point label distribution")
    print(SEP)
    meanings = {
        0: "Both No-PCL  (0+0)",
        1: "Borderline    (0+1)",
        2: "Both Borderline (1+1)",
        3: "One clear     (1+2)",
        4: "Both clear PCL (2+2)",
    }
    print(f"{'Lbl':<5}  {'Count':>7}  {'%':>6}  Annotation Meaning")
    print("-" * 55)
    for lbl in range(5):
        c = counts5[lbl]
        print(f"{lbl:<5}  {c:>7,}  {pct(c,total):>5.1f}%  {meanings[lbl]}")
    print(f"{'All':<5}  {total:>7,}  100.0%")
    print()

    print(SEP)
    print("TABLE 2: Binary distribution")
    print(SEP)
    print(f"  No-PCL  {{0,1}} : {n_neg:>6,}  ({pct(n_neg,total):.2f}%)")
    print(f"  PCL     {{2,3,4}}: {n_pos:>6,}  ({pct(n_pos,total):.2f}%)")
    print(f"  Total         : {total:>6,}")
    print(f"  Imbalance ratio (neg:pos) = {n_neg/n_pos:.2f}:1")
    print()

    print(SEP)
    print("TABLE 3: Sequence-length statistics (word count)")
    print(SEP)
    for name, arr in [("No-PCL", wc_neg), ("PCL", wc_pos), ("Overall", wc_all)]:
        print(f"  {name:<10}  n={len(arr):>6,}  "
              f"mean={arr.mean():.1f}  "
              f"median={np.median(arr):.0f}  "
              f"std={arr.std():.1f}  "
              f"max={arr.max()}  "
              f"p95={np.percentile(arr,95):.0f}  "
              f"p99={np.percentile(arr,99):.0f}")
    print()

    print(SEP)
    print("TABLE 4: Per-keyword PCL rate (sorted by rate desc.)")
    print(SEP)
    kw_total = Counter(keywords)
    kw_pos   = Counter(r["keyword"] for r in data if r["binary"] == 1)
    kw_rows  = sorted(kw_total.keys(),
                      key=lambda k: kw_pos.get(k,0)/kw_total[k], reverse=True)
    print(f"  {'Keyword':<16}  {'Total':>6}  {'PCL':>5}  {'Rate':>6}")
    print("  " + "-" * 38)
    for kw in kw_rows:
        tot = kw_total[kw]; pos = kw_pos.get(kw, 0)
        print(f"  {kw:<16}  {tot:>6,}  {pos:>5}  {pct(pos,tot):>5.1f}%")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 – Class Distribution  (EDA Technique 1)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # Panel (a): 5-point distribution
    ax = axes[0]
    x5 = list(range(5))
    c5 = [counts5[i] for i in x5]
    bars = ax.bar(x5, c5, color=PALETTE_5, edgecolor="white",
                  linewidth=0.9, zorder=3, width=0.65)
    ax.set_xticks(x5)
    ax.set_xticklabels([f"Label {i}" for i in x5])
    ax.set_ylabel("Paragraphs")
    ax.set_title("(a) 5-Point Label Distribution")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for bar, c in zip(bars, c5):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 60,
                f"{c:,}", ha="center", va="bottom", fontsize=8)
    # PCL shading
    ax.axvspan(1.55, 4.45, color="#ffd6d6", alpha=0.25, zorder=0)
    ax.text(3.5, max(c5)*0.88, "← PCL zone",
            ha="center", fontsize=8, color="#a00000", style="italic")
    # Legend patches
    no_pcl_patch = mpatches.Patch(color=PALETTE_5[0], alpha=0.6, label="No-PCL (0)")
    brd_patch    = mpatches.Patch(color=PALETTE_5[1], alpha=0.6, label="Borderline (1)")
    pcl_patch    = mpatches.Patch(color=PALETTE_5[3], alpha=0.8, label="PCL (2–4)")
    ax.legend(handles=[no_pcl_patch, brd_patch, pcl_patch],
              loc="upper right", fontsize=7.5, framealpha=0.8)

    # Panel (b): Binary distribution with imbalance annotation
    ax = axes[1]
    labels_b  = ["No PCL\n(labels 0–1)", "PCL\n(labels 2–4)"]
    counts_b  = [n_neg, n_pos]
    colors_b  = [C_NEG, C_POS]
    bars2 = ax.bar(labels_b, counts_b, color=colors_b, width=0.40,
                   edgecolor="white", linewidth=0.9, zorder=3)
    ax.set_ylabel("Paragraphs")
    ax.set_title("(b) Binary Class Distribution")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for bar, c in zip(bars2, counts_b):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 100,
                f"{c:,}\n({pct(c,total):.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, n_neg * 1.20)
    # Imbalance annotation arrow
    ax.annotate(
        f"Ratio\n{n_neg/n_pos:.1f}:1",
        xy=(0.5, (n_neg+n_pos)/2),
        xytext=(0.5, n_neg * 0.6),
        ha="center", fontsize=8.5, color="#555",
        arrowprops=dict(arrowstyle="<->", color="#aaa", lw=1.0),
    )

    fig.tight_layout(pad=1.8)
    out1 = FIG_DIR / "fig1_class_distribution.pdf"
    fig.savefig(out1, bbox_inches="tight", format="pdf")
    print(f"Saved → {out1}")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 – Sequence Length Analysis  (EDA Technique 2)
    # ═══════════════════════════════════════════════════════════════════════════
    CLIP = 500   # clip tail for histogram readability

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # Panel (a): Overlapping density histograms
    ax = axes[0]
    bins = np.linspace(0, CLIP, 51)
    ax.hist(np.clip(wc_neg, 0, CLIP), bins=bins, density=True,
            alpha=0.50, color=C_NEG, label=f"No PCL  (n={n_neg:,})", zorder=3)
    ax.hist(np.clip(wc_pos, 0, CLIP), bins=bins, density=True,
            alpha=0.65, color=C_POS, label=f"PCL  (n={n_pos:,})", zorder=4)
    ax.axvline(wc_neg.mean(), color=C_NEG, linestyle="--", linewidth=1.4,
               label=f"Mean No-PCL = {wc_neg.mean():.0f}w")
    ax.axvline(wc_pos.mean(), color=C_POS, linestyle="--", linewidth=1.4,
               label=f"Mean PCL = {wc_pos.mean():.0f}w")
    ax.axvline(128, color="gray", linestyle=":", linewidth=1.0,
               label="128-token ref.")
    ax.set_xlabel("Word Count (clipped at 500)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Length Density by Class")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Panel (b): Side-by-side box plots
    ax = axes[1]
    bp = ax.boxplot(
        [wc_neg, wc_pos],
        tick_labels=["No PCL\n{0\u20131}", "PCL\n{2\u20134}"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        flierprops=dict(marker=".", markersize=1.8, alpha=0.25),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        whis=[5, 95],
        widths=0.45,
    )
    bp["boxes"][0].set_facecolor(C_NEG + "55")
    bp["boxes"][0].set_edgecolor(C_NEG)
    bp["boxes"][1].set_facecolor(C_POS + "55")
    bp["boxes"][1].set_edgecolor(C_POS)

    # Annotate medians
    for i, (arr, clr) in enumerate([(wc_neg, C_NEG), (wc_pos, C_POS)], 1):
        med = np.median(arr)
        ax.text(i + 0.27, med, f"med={med:.0f}", va="center",
                fontsize=7.5, color=clr, fontweight="bold")

    ax.set_ylabel("Word Count")
    ax.set_title("(b) Box Plots (whiskers = 5th–95th pct.)")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=1.8)
    out2 = FIG_DIR / "fig2_sequence_length.pdf"
    fig.savefig(out2, bbox_inches="tight", format="pdf")
    print(f"Saved → {out2}")
    plt.close(fig)

    print("\nPhase 2 EDA complete.")


if __name__ == "__main__":
    main()
