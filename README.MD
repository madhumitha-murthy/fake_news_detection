# Fake News Detection in Dravidian Languages

**Published research · DravidianLangTech @ EACL 2024**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Conference](https://img.shields.io/badge/EACL_2024-DravidianLangTech-blueviolet?style=flat-square)](https://aclanthology.org/2024.dravidianlangtech-1.25)
[![Task 1 Rank](https://img.shields.io/badge/Task_1_Rank-%233_of_all_teams-gold?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

Transformer-based fake news detection for **Malayalam**, a low-resource Dravidian language. Fine-tuned XLM-RoBERTa and mBERT on two classification tasks: binary fake detection and five-class misinformation categorisation. The system placed **3rd overall** at the DravidianLangTech@EACL 2024 shared task.

> **TechWhiz@DravidianLangTech 2024: Fake News Detection Using Deep Learning Models**
> Madhumitha M, Kunguma Akshatra M, Tejashri J, C Jerin Mahibha
> Meenakshi Sundararajan Engineering College, Chennai, India
> *Proceedings of the Fourth Workshop on Speech, Vision, and Language Technologies for Dravidian Languages, EACL 2024, pp. 200–204*

---

## Table of Contents

- [Tasks](#tasks)
- [Results](#results)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Citation](#citation)

---

## Tasks

### Task 1 — Binary Classification
Classify Malayalam social media posts as **Fake** or **Original**.

### Task 2 — Multi-class Classification
Classify Malayalam news articles into one of five misinformation categories:

| Label | Description |
|---|---|
| `False` | Entirely fabricated claim |
| `Mostly False` | Predominantly incorrect with minor truths |
| `Half True` | Partially accurate but missing key context |
| `Partly False` | Mostly accurate with some false elements |
| `Mostly True` | Accurate with minor inaccuracies |

---

## Results

### Task 1 — Binary Fake News Detection

| Model | Macro F1 | Accuracy | Leaderboard Rank |
|---|---|---|---|
| **XLM-RoBERTa** | **0.84** | **0.84** | **3rd** |
| mBERT | 0.84 | 0.84 | — |
| ALBERT | 0.56 | 0.58 | — |

**Per-class breakdown (XLM-RoBERTa on dev set):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Original | 0.84 | 0.85 | 0.84 | 409 |
| Fake | 0.85 | 0.83 | 0.84 | 406 |

![Task 1 XLM-RoBERTa](task1/results/task1%20xlmroberta.png)

### Task 2 — Multi-class Fake Category Detection

| Model | Macro F1 | Accuracy |
|---|---|---|
| **XLM-RoBERTa** | **0.21** | **0.74** |
| mBERT | 0.16 | 0.63 |
| BERT (English) | 0.11 | 0.25 |

The low macro F1 reflects the extreme class imbalance (1,246 "False" vs 1 "Mostly True") rather than model failure — accuracy of 0.74 confirms meaningful learning.

![Task 2 XLM-RoBERTa](task2/results/task2xlmroberta.png)

---

## Model Architecture

Both tasks share the same fine-tuning pipeline. Only the classification head width (`K`) and training data differ.

```
Raw Text (Malayalam social media / news)
         │
         ▼
┌──────────────────────────┐
│       Preprocessing       │
│  • Remove URLs            │
│  • Remove @mentions       │
│  • Remove #hashtags       │
│  • Strip punctuation      │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│           Transformer Encoder                 │
│                                               │
│  XLM-RoBERTa-base          mBERT              │
│  ─────────────────         ──────────────     │
│  12 layers                 12 layers          │
│  768 hidden units          768 hidden units   │
│  12 attention heads        12 attention heads │
│  250k vocab (100 langs)    120k vocab (104 l) │
└──────────────────────────────────────────────┘
         │  [CLS] token representation (768-d)
         ▼
┌──────────────────────────┐
│    Classification Head    │
│   Dropout → Linear(768,K)│
│   K = 2  (Task 1)        │
│   K = 5  (Task 2)        │
└──────────────────────────┘
         │
         ▼
    Softmax → Predicted label
```

**Training configuration:**

| Hyperparameter | Task 1 | Task 2 |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 2e-5 (Trainer default) | 1e-5 |
| Batch size | 8 | 16 |
| Epochs | 3 | 3 |
| Max sequence length | 256 | 256 |
| Training framework | HuggingFace Trainer | Custom PyTorch loop |

**Why XLM-RoBERTa?**
Pre-trained on 2.5TB of filtered CommonCrawl data across 100 languages using a joint BPE vocabulary of 250,002 tokens. Its strong cross-lingual representations outperform language-specific models on low-resource tasks, making it the best fit for Malayalam — a morphologically rich Dravidian language with limited labelled data.

---

## Dataset

Provided by the shared task organisers and **not included in this repository**. To reproduce results, request the data from the [DravidianLangTech@EACL 2024 shared task page](https://sites.google.com/view/dravidianlangtech-2024) and place files as follows:

```
task1/data/train.csv
task1/data/dev.csv
task1/data/test.csv

task2/data/train_data_mal_fake_detect.csv
task2/data/test-data.csv
```

**Task 1 — Data distribution:**

| Split | Fake | Original | Total |
|---|---|---|---|
| Train | 1,599 | 1,658 | 3,257 |
| Dev | 406 | 409 | 815 |
| Test | — | — | 1,019 |

**Task 2 — Training distribution (severely imbalanced):**

| Category | Count | % of total |
|---|---|---|
| False | 1,246 | 73.9% |
| Mostly False | 239 | 14.2% |
| Half True | 141 | 8.4% |
| Partly False | 42 | 2.5% |
| Mostly True | 1 | 0.06% |

The imbalance is addressed with stratified train/val splitting and oversampled balanced mini-batches during training (200 samples per class with replacement).

---

## Project Structure

```
fake-news-detection/
│
├── app.py                        # Streamlit web demo
├── requirements.txt
│
├── notebooks/
│   ├── Task1.ipynb               # Interactive binary classification workflow
│   └── Task2.ipynb               # Interactive multi-class workflow
│
├── task1/                        # Binary fake detection (Fake / Original)
│   ├── data/                     # CSVs — git-ignored, see Dataset section
│   ├── models/                   # Saved checkpoints — git-ignored
│   ├── outputs/                  # Submission CSVs — git-ignored
│   ├── results/                  # Evaluation heatmaps and confusion matrices
│   └── src/
│       ├── preprocess.py         # Text cleaning
│       ├── train.py              # Fine-tune XLM-RoBERTa via HuggingFace Trainer
│       ├── evaluate.py           # Evaluate on dev set, print classification report
│       └── predict.py            # Generate test-set predictions; exposes predict_single()
│
├── task2/                        # Multi-class fake category detection (5 classes)
│   ├── data/                     # CSVs — git-ignored
│   ├── models/                   # Saved checkpoints — git-ignored
│   ├── outputs/                  # Submission CSVs — git-ignored
│   ├── results/                  # Evaluation heatmaps and confusion matrices
│   └── src/
│       ├── config.py             # Shared MODEL_MAP (mbert, xlmr)
│       ├── preprocess.py         # Split, label-encode, balanced oversampling
│       ├── train.py              # Fine-tune via custom PyTorch loop
│       ├── evaluate.py           # Evaluation with confusion matrix and heatmap output
│       └── predict.py            # Batch inference on test set
│
└── paper and conference/
    └── Paper.pdf                 # Published paper (EACL 2024)
```

---

## Installation

```bash
git clone https://github.com/madhumitha-murthy/Fake-News-Detection-in-Dravidian-Languages-
cd Fake-News-Detection-in-Dravidian-Languages-

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Option A — Jupyter Notebooks (recommended for exploration)

```bash
jupyter notebook "notebooks /Task1.ipynb"   # binary classification
jupyter notebook "notebooks /Task2.ipynb"   # multi-class classification
```

### Option B — Scripts (recommended for reproducibility)

**Task 1:**

```bash
cd task1/src

python train.py       # fine-tunes XLM-RoBERTa; saves to task1/models/
python evaluate.py    # prints classification report on dev.csv
python predict.py     # generates task1/outputs/xlm-roberta-base_submission.csv
```

**Task 2:**

```bash
cd task2/src

python preprocess.py              # split, encode, and balance the training data
python train.py --model xlmr     # fine-tune XLM-RoBERTa  (or: --model mbert)
python evaluate.py --model xlmr  # print report + save PNG visualisations
python predict.py --model xlmr   # generate task2/outputs/task2_xlmr_predictions.csv
```

Available training flags for Task 2:

```bash
python train.py --model xlmr --epochs 5 --batch_size 32 --lr 2e-5
```

### Option C — Streamlit Web App

```bash
streamlit run app.py
```

The app loads the Task 1 XLM-RoBERTa checkpoint from `task1/models/xlm-roberta-base/`. Train the model first if the checkpoint does not exist.

### Programmatic inference (Task 1)

```python
import sys
sys.path.insert(0, "task1/src")
from predict import predict_single

# Malayalam text example
text = "നല്ല അവതരണം. സത്യം പുറത്തു വരട്ടെ"
result = predict_single(text)
print(result)  # → "Fake" or "Original"
```

---

## Key Findings

- **XLM-RoBERTa and mBERT tied on Task 1** (F1 0.84) but XLM-RoBERTa was selected for its superior cross-lingual representations on unseen language patterns.
- **Task 2 is a data problem more than a modelling problem.** The extreme class imbalance (73.9% "False") pushes all models toward majority-class predictions. Oversampling minority classes improved stability but macro F1 remained low across all models.
- **English-only BERT is unsuitable for Malayalam.** Its vocabulary has near-zero overlap with Dravidian script tokens, producing random embeddings.
- **ALBERT underperformed significantly on Task 1** (F1 0.56 vs 0.84). Its parameter-sharing mechanism appears less effective for morphologically rich languages with complex inflectional patterns.

---

## Limitations

- Textual modality only — multimodal misinformation (images, video, audio) is out of scope.
- Models inherit biases from their multilingual pre-training corpora.
- Task 2 remains largely unsolved due to the extreme label scarcity for minority classes.
- No cross-lingual transfer experiments to other Dravidian languages (Tamil, Telugu, Kannada).

---

## Citation

```bibtex
@inproceedings{madhumitha2024techwhiz,
  title     = {{TechWhiz@DravidianLangTech} 2024: Fake News Detection Using Deep Learning Models},
  author    = {Madhumitha M and Kunguma Akshatra M and Tejashri J and C Jerin Mahibha},
  booktitle = {Proceedings of the Fourth Workshop on Speech, Vision, and Language
               Technologies for Dravidian Languages},
  pages     = {200--204},
  year      = {2024},
  month     = {March},
  url       = {https://aclanthology.org/2024.dravidianlangtech-1.25}
}
```

---

## Authors

**Madhumitha M** · **Kunguma Akshatra M** · **Tejashri J** · **C Jerin Mahibha**
Meenakshi Sundararajan Engineering College, Chennai, India
