# NLP Coursework 2026 — PCL Detection

**Course:** 70016 Natural Language Processing — Imperial College London
**Leaderboard Name:** ak7025
**Deadline:** Wednesday, 4th March 2026, 7:00 PM
**Task:** Detect Patronising and Condescending Language (PCL) in news paragraphs
**Metric:** F1 score of the positive class (PCL = 1)
**Baseline:** RoBERTa-base — Dev F1 = 0.48 | Test F1 = 0.49

---

## Repository Structure

```
.
├── README.md                          
│
├── BestModel/
│   ├── train.py                       ← RoBERTa-large fine-tuning pipeline
│   ├── predict.py                     ← inference from a saved checkpoint
│   ├── baseline_cpu.py                ← CPU-runnable TF-IDF + LR baseline
│   └── requirements.txt               ← Python dependencies
│
├── Dont_Patronize_Me_Trainingset/
│   ├── dontpatronizeme_pcl.tsv        ← training set 
│   ├── dontpatronizeme_categories.tsv ← span-level PCL category labels
│   └── README.txt                     
│
├── figures/
│   ├── fig1_class_distribution.pdf    ← EDA: class imbalance
│   ├── fig2_sequence_length.pdf       ← EDA: sequence length by class
│   ├── fig3_confusion_matrix.pdf      ← Evaluation: confusion matrix (RoBERTa-large)
│   └── fig4_error_breakdown.pdf       ← Evaluation: keyword errors + ablation
│
├── eda.py                             ← EDA script (generates figures/)
├── verify_labels.py                   ← label distribution check
├── error_analysis.py                  ← error analysis + ablation study
│
├── dev.txt                            ← dev predictions (1047 lines, internal split)
└── test.txt                           ← test predictions (3832 lines, official test set)
```

---

## Proposed Approach (BestModel)

Four components over the RoBERTa-base baseline:

| # | Component | Description |
|---|---|---|
| C1 | RoBERTa-large backbone | `roberta-large` — 355M params, 24 layers, 2.8× the baseline capacity |
| C2 | Class-weighted BCE loss | `pos_weight = 9.54` (empirical 9.54:1 imbalance ratio) |
| C3 | Keyword-prefix input | `"Community: {keyword} {text}"` — community-type conditioning signal |
| C4 | Dev-set threshold tuning | Sweep τ ∈ [0.10, 0.90], step 0.01, to maximise positive-class F1 |

> **Note:** The original proposal used `microsoft/deberta-v3-base` for C1. This was
> substituted with `roberta-large` due to a `CUBLAS_STATUS_INVALID_VALUE` error in
> cuBLAS's strided batched GEMM kernel — DeBERTa-v3's disentangled attention is
> incompatible with the GPU/cuBLAS version on the available hardware (affects both v2
> and v3). `roberta-large` achieves strong results and uses the same fine-tuning pipeline.

---

## Reproducing Results

> **GPU required** for `train.py`. Trained on Google Colab (NVIDIA T4 GPU).
> The official SemEval dev/test TSV files were not publicly available; dev results
> use an internal 90/10 training split (`--split_dev`). Test predictions were
> generated from the official test file using `predict.py`.

### 1. Install dependencies

```bash
pip install -r BestModel/requirements.txt
```

### 2. Verify training-set label distribution

```bash
python3 verify_labels.py
```

### 3. Run EDA (generates figures/)

```bash
python3 eda.py
```

### 4. Train RoBERTa-large on Google Colab (GPU)

In a new Colab notebook (Runtime → T4 GPU):

```python
!git clone https://github.com/thekhananas/ak-nlp-fishname.git
%cd ak-nlp-fishname
!pip install -r BestModel/requirements.txt -q

!python BestModel/train.py \
  --train_file Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv \
  --output_dir BestModel/checkpoints \
  --split_dev \
  --dev_out    dev.txt \
  --test_out   test.txt
```

This saves the best checkpoint to `BestModel/checkpoints/best_model/` and writes `dev.txt`.

**With official dev + test files** (if available):

```bash
python BestModel/train.py \
  --train_file Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv \
  --dev_file   <path/to/dev_semeval22_task4.tsv> \
  --test_file  <path/to/test_semeval22_task4.tsv> \
  --output_dir BestModel/checkpoints
```

### 5. Generate test.txt from official test file (inference only)

Used to produce the `test.txt` in this repository:

```bash
python BestModel/predict.py \
  --model_dir  BestModel/checkpoints/best_model \
  --input_file task4_test.tsv \
  --output     test.txt
```

`predict.py` automatically loads the saved threshold (τ = 0.77) from the checkpoint.

### 6. CPU baseline (no GPU required)

```bash
python BestModel/baseline_cpu.py \
  --train_file Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv \
  --dev_out    dev.txt \
  --test_out   test.txt \
  --test_lines 3832
```

### 7. Run error analysis (generates figures/)

```bash
python3 error_analysis.py
```

---

## Prediction File Format

`dev.txt` and `test.txt` contain one binary prediction per line:

```
0       ← No PCL
1       ← PCL
0
...
```

- `dev.txt` — 1047 lines, predictions on internal 10% hold-out split
- `test.txt` — **exactly 3832 lines**, predictions on the official test set

---

## Key Results

| Model | Dev F1 | P | R | Notes |
|---|---|---|---|---|
| Baseline (RoBERTa-base) | 0.48 | — | — | SemEval organisers |
| TF-IDF + LR (C2+C3) | 0.46 | 0.40 | 0.54 | `baseline_cpu.py`, internal split |
| **RoBERTa-large (C1–C4)** | **0.6067** | **0.69** | **0.54** | `train.py`, Colab T4, internal split, τ=0.77 |

---

## Citation

> Pérez-Almendros, C., Espinosa-Anke, L., & Schockaert, S. (2020).
> *Don't Patronize Me! An Annotated Dataset with Patronising and Condescending
> Language towards Vulnerable Communities.*
> Proceedings of COLING 2020, pp. 5891–5902.
