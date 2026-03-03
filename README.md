# NLP Coursework 2026 — PCL Detection

**Course:** 70016 Natural Language Processing — Imperial College London

---

## Repository Structure

```
.
├── README.md                          
├── report.pdf                         
├── report.tex                         
│
├── BestModel/
│   ├── train.py                       
│   ├── predict.py                     
│   ├── baseline_cpu.py                
│   └── requirements.txt               
│
├── Dont_Patronize_Me_Trainingset/
│   ├── dontpatronizeme_pcl.tsv        
│   ├── dontpatronizeme_categories.tsv 
│   └── README.txt                     
│
├── figures/
│   ├── fig1_class_distribution.pdf    
│   ├── fig2_sequence_length.pdf       
│   ├── fig3_confusion_matrix.pdf      
│   └── fig4_error_breakdown.pdf       
│
├── eda.py                             
├── verify_labels.py                   
├── error_analysis.py                  
│
├── dev.txt                            ← dev set predictions (one 0/1 per line)
└── test.txt                           ← test set predictions (3832 lines)
```

---

## Proposed Approach (BestModel)

Four components over the RoBERTa-base baseline:

| # | Component | Description |
|---|---|---|
| C1 | DeBERTa-v3-base backbone | `microsoft/deberta-v3-base` — disentangled attention + EMD |
| C2 | Class-weighted BCE loss | `pos_weight = 9.54` (empirical imbalance ratio) |
| C3 | Keyword-prefix input | `"Community: {keyword} {text}"` |
| C4 | Dev-set threshold tuning | Sweep τ ∈ [0.10, 0.90] to maximise positive-class F1 |

---

## Reproducing Results

### 1. Install dependencies

```bash
pip install -r BestModel/requirements.txt
```

> **GPU required** for `train.py`. Tested on Google Colab T4/A100.

### 2. Verify training-set label distribution

```bash
python3 verify_labels.py
```

### 3. Run EDA (generates figures/)

```bash
python3 eda.py
```

### 4. Train DeBERTa-v3-base (GPU)

```bash
python BestModel/train.py \
  --train_file Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv \
  --dev_file   <path/to/dev_semeval22_task4.tsv> \
  --test_file  <path/to/test_semeval22_task4.tsv> \
  --output_dir BestModel/checkpoints
# Writes dev.txt and test.txt automatically
```

**Training-only mode** (internal 90/10 split as dev):

```bash
python BestModel/train.py \
  --train_file Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv \
  --output_dir BestModel/checkpoints \
  --split_dev
```

### 5. Inference from saved checkpoint

```bash
python BestModel/predict.py \
  --model_dir  BestModel/checkpoints/best_model \
  --input_file <path/to/input.tsv> \
  --output     predictions.txt
```

### 6. CPU baseline (no GPU / no official dev-test files)

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

- `dev.txt` — one line per example in the official dev set
- `test.txt` — **exactly 3832 lines** (one per test-set example)


---

## Key Results

| Model | Dev F1 | Notes |
|---|---|---|
| Baseline (RoBERTa-base) | 0.48 | SemEval organisers |
| TF-IDF + LR (C2+C3, internal split) | 0.46 | `baseline_cpu.py` |
| **DeBERTa-v3-base (C1–C4)** | **target ≥ 0.57** | `train.py`, requires GPU |

---

## Citation

> Pérez-Almendros, C., Espinosa-Anke, L., & Schockaert, S. (2020).
> *Don't Patronize Me! An Annotated Dataset with Patronising and Condescending
> Language towards Vulnerable Communities.*
> Proceedings of COLING 2020, pp. 5891–5902.
