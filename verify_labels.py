"""
This is a verification script for the DPM, training set label distribution.
Reads dontpatronizeme_pcl.tsv and reports:
  - Total paragraph count
  - Per-label counts (0–4)
  - Binary split: No-PCL {0,1} vs PCL {2,3,4}
  - Class imbalance ratio
"""

import csv
from collections import Counter
from pathlib import Path

TSV_PATH = Path(__file__).parent / "Dont_Patronize_Me_Trainingset" / "dontpatronizeme_pcl.tsv"

def load_labels(path: Path) -> list[int]:
    labels = []
    with open(path, encoding="utf-8") as f:
        # Skip the 4-line disclaimer header (lines 1-4)
        for _ in range(4):
            next(f)
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue          # skip malformed lines
            try:
                labels.append(int(row[5]))
            except ValueError:
                continue          # skip header/comment rows
    return labels


def main() -> None:
    labels = load_labels(TSV_PATH)

    counts = Counter(labels)
    total = len(labels)

    print("=" * 50)
    print("DPM! Training Set — Label Distribution")
    print("=" * 50)
    print(f"\n{'Label':<10} {'Count':>8}  {'%':>7}")
    print("-" * 30)
    for lbl in sorted(counts):
        pct = 100 * counts[lbl] / total
        print(f"{lbl:<10} {counts[lbl]:>8}  {pct:>6.2f}%")

    no_pcl = sum(counts[k] for k in [0, 1])
    pcl    = sum(counts[k] for k in [2, 3, 4])

    print("\n" + "-" * 30)
    print(f"{'No-PCL {0,1}':<20} {no_pcl:>8}  {100*no_pcl/total:>6.2f}%")
    print(f"{'PCL    {2,3,4}':<20} {pcl:>8}  {100*pcl/total:>6.2f}%")
    print(f"{'TOTAL':<20} {total:>8}")
    print(f"\nImbalance ratio  (neg:pos) = {no_pcl/pcl:.2f}:1")
    print("=" * 50)


if __name__ == "__main__":
    main()
