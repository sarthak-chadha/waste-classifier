# ♻️ Waste Classifier AI: Deep Transfer Learning & Deployment

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.26+-EE4C2C.svg)](https://pytorch.org/vision/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.39%25-brightgreen.svg)]()
[![Macro F1](https://img.shields.io/badge/Macro--F1-0.9710-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end computer vision pipeline and production deployment for automated municipal waste classification. 

Using **EfficientNet-B4** fine-tuned on standardized 384×384 imagery across **10 waste categories**, the model achieves **97.39% test accuracy** and **0.9710 Macro-F1** on 1,840 strictly held-out test photos, cutting classification errors nearly in half compared to a frozen ImageNet baseline.

---

## 📌 Table of Contents

- [Benchmark Results & Scientific Proof](#-benchmark-results--scientific-proof)
- [Per-Class Performance](#-per-class-performance)
- [Inference Latency & Throughput](#-inference-latency--throughput)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Reproducibility & Execution Guide](#-reproducibility--execution-guide)
- [Web Application & Deployment](#-web-application--deployment)

---

## 🏆 Benchmark Results & Scientific Proof

To conclusively isolate the effect of representation learning from training duration, the experiment included a controlled 3-way evaluation:
1. **Base (Head-only)**: Pretrained ImageNet backbone frozen, training only the 10-class linear head (10 epochs).
2. **Control (Extended Frozen)**: Continued training the frozen head for 15 additional epochs (25 total frozen epochs) using the exact same oversampled batches and learning rate schedule as fine-tuning.
3. **Fine-Tuned (Full Network)**: Started from the best base checkpoint, unfreezing all convolutional layers with differential learning rates (`1e-4` backbone, `1e-3` head) for 15 epochs.

### Overall Test Performance ($n = 1,840$ held-out images)

| Model | Backbone | Trainable Params | Test Accuracy (95% CI) | Test Macro-F1 (95% CI) | Weighted-F1 | Test Errors |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base** | Frozen | 17.9 K (0.1%) | 95.16% [94.18%, 96.03%] | 0.9489 [0.9374, 0.9586] | 0.9515 | 89 / 1840 |
| **Control** | Frozen | 17.9 K (0.1%) | 95.65% [94.67%, 96.58%] | 0.9545 [0.9435, 0.9644] | 0.9566 | 80 / 1840 |
| **Fine-Tuned** | **Unfrozen** | **17.57 M (100%)** | **97.39% [96.63%, 98.10%]** | **0.9710 [0.9617, 0.9791]** | **0.9738** | **48 / 1840** |

### Statistical Hypothesis Testing (Paired Bootstrap & Exact McNemar)

| Comparison | $\Delta$ Macro-F1 (95% Bootstrap CI) | Only Model A Correct | Only Model B Correct | McNemar $p$-value | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control vs. Base** | +0.0056 [−0.0015, +0.0129] | 22 | 13 | $p = 0.175$ | **NO** ($p > 0.05$, CI spans zero) |
| **Fine-Tuned vs. Control** | **+0.0165 [+0.0076, +0.0259]** | **42** | **10** | **$p = 9.06 \times 10^{-6}$** | **YES** ($p < 0.00001$) |
| **Fine-Tuned vs. Base** | **+0.0222 [+0.0122, +0.0326]** | **52** | **11** | **$p = 1.67 \times 10^{-7}$** | **YES** ($p < 0.000001$) |

> **Key Empirical Finding:** Extra epochs alone on a frozen backbone (**Control vs. Base**) showed **no statistically significant gain** ($p = 0.175$). Unfreezing the backbone (**Fine-Tuned vs. Control**) delivered an overwhelmingly significant leap ($p < 10^{-5}$), proving that the performance boost is **100% genuine domain representation learning**.

---

## 🎯 Per-Class Performance

All 10 classes exhibited positive performance gains from end-to-end representation adaptation:

| Class | Test Images ($n$) | Base F1 | Control F1 | Fine-Tuned F1 | $\Delta$ Gain vs Base | Precision | Recall | Primary Disposal Stream |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **clothes** | 284 | 0.993 | 0.993 | **1.000** | +0.007 | 100.0% | 100.0% | Textile recycling / donation |
| **shoes** | 218 | 0.991 | 0.993 | **0.995** | +0.005 | 99.5% | 99.5% | Shoe bank / reuse |
| **battery** | 114 | 0.969 | 0.969 | **0.987** | +0.018 | 99.1% | 98.2% | Hazardous E-waste drop-off |
| **organic** | 105 | 0.976 | 0.986 | **0.986** | +0.010 | 98.1% | 99.1% | Compost / green bio-bin |
| **cardboard** | 211 | 0.938 | 0.955 | **0.974** | **+0.036** | 97.6% | 97.2% | Clean paper & cardboard |
| **metal** | 139 | 0.903 | 0.910 | **0.968** | **+0.065** | 95.1% | 98.6% | Mixed metal recycling |
| **glass** | 261 | 0.936 | 0.934 | **0.966** | **+0.030** | 95.9% | 97.3% | Glass recycling |
| **paper** | 201 | 0.943 | 0.953 | **0.961** | +0.018 | 94.7% | 97.5% | Dry paper stream |
| **plastic** | 239 | 0.921 | 0.927 | **0.949** | **+0.027** | 97.4% | 92.5% | Mixed plastic container bin |
| **trash** | 68 | 0.919 | 0.926 | **0.925** | +0.007 | 93.9% | 91.2% | General non-recyclable landfill |

* Visually ambiguous categories benefited the most: **metal** (+6.5% F1), **cardboard** (+3.6% F1), **glass** (+3.0% F1), and **plastic** (+2.7% F1).
* **clothes** achieved perfect accuracy: **284 out of 284 images correctly classified**.

---

## ⚡ Inference Latency & Throughput

Benchmarked on an **NVIDIA GeForce RTX 5060 Laptop GPU** (384×384 input resolution, `bfloat16` precision, forward pass only):

| Metric | Benchmark Result | Practical Real-World Capability |
| :--- | :---: | :--- |
| **Single Image Latency (Median)** | **14.45 ms** | Real-time single-camera stream (~69 FPS) |
| **Single Image Latency (95th %ile)** | **22.50 ms** | Low jitter for robotic picking |
| **Batched Throughput (Batch 16)** | **263.8 img / sec** | Industrial conveyor sorting (>15,000 items/min) |

---

## 🛠️ Pipeline Architecture

The pipeline is organized into modular, reproducible steps:

```
                  ┌────────────────────────┐
                  │ 0. sanity_check.py     │ ── GPU, environment & dataset validation
                  └───────────┬────────────┘
                              ▼
                  ┌────────────────────────┐
                  │ 1. step1_split_data.py │ ── Stratified split (70/15/15) + oversampling plan
                  └───────────┬────────────┘
                              ▼
                  ┌────────────────────────┐
                  │ 2. step2_dataset.py    │ ── Online dynamic oversampling & augmentations
                  └───────────┬────────────┘
                              ▼
                  ┌────────────────────────┐
                  │ 3. step3_train.py      │ ── Base, Control & Fine-Tuning in bf16
                  └───────────┬────────────┘
                              ▼
                  ┌────────────────────────┐
                  │ 4. step4_evaluate.py   │ ── Paired bootstrap CI, McNemar & error analysis
                  └───────────┬────────────┘
                              ▼
                  ┌────────────────────────┐
                  │ 5. app/app.py          │ ── Streamlit interactive deployment
                  └────────────────────────┘
```

1. **`src/sanity_check.py`**: Validates GPU compute architecture, available VRAM, cuDNN, package versions, image file corruption, exact resolutions, color modes, and hash duplicate detection.
2. **`src/step1_split_data.py`**: Performs stratified 70/15/15 partitioning and calculates a per-epoch oversampling target (`target = max(n, min(avg, 2n))`) to handle minority class imbalance without writing duplicated files to disk.
3. **`src/step2_dataset.py`**: Custom `EpochOversampler` redraws fresh minority-class samples without replacement each epoch and pairs them with dynamic geometric augmentations (vertical/horizontal flips, color jitter, Gaussian blur, and ImageNet mean rotation padding).
4. **`src/step3_train.py`**: Implements mixed-precision (`bfloat16`) training using AdamW, cosine annealing with linear warmup, and gradient accumulation. Executes Base, Control, and Fine-Tuning under identical epoch seeds.
5. **`src/step4_evaluate.py`**: Evaluates all three checkpoints on the held-out test split, reproducing validation sanity checks before calculating paired bootstrap 95% confidence intervals, exact McNemar test statistics, confusion matrix heatmaps, and failure galleries.
6. **`app/`**: Standalone, containerized Streamlit application featuring aspect-ratio-preserving letterboxing, transparent PNG alpha-compositing, EXIF orientation correction, top-3 confidence bars, and customized municipal disposal recommendations.

---

## 📁 Project Directory Structure

```
waste_classifier/
├── .gitignore                         # Excludes raw data, vfl, checkpoints, and cache
├── README.md                          # Main project documentation
├── requirements.txt                   # Root environment dependencies
├── data/                              # Dataset root (10 class subdirectories)
│   ├── battery/
│   ├── cardboard/
│   └── ...
├── outputs/                           # Generated project artifacts
│   ├── splits.csv                     # Deterministic train/val/test assignments
│   ├── oversampling_plan.json         # Per-class target plan & class_to_idx mapping
│   ├── runs/                          # Checkpoints, logs, and curves
│   │   ├── base/                      # (best.pt, curves.png, log.csv, summary.json)
│   │   ├── control/                   # (best.pt, curves.png, log.csv, summary.json)
│   │   └── finetune/                  # (best.pt, curves.png, log.csv, summary.json)
│   └── evaluation/                    # Final test set evaluation deliverables
│       ├── overall_comparison.csv     # Metrics with 95% bootstrap CIs
│       ├── per_class_metrics.csv      # Precision, recall, and F1 across all classes
│       ├── predictions.csv            # Per-image prediction logs
│       ├── results.json               # Full machine-readable evaluation report
│       ├── confusion_matrices.png     # Side-by-side normalized confusion matrices
│       ├── per_class_f1.png           # 10-class grouped bar chart
│       └── misclassifications.png     # Visual gallery of top-confidence errors
├── src/                               # Pipeline source scripts
│   ├── sanity_check.py
│   ├── step1_split_data.py
│   ├── step2_dataset.py
│   ├── step3_train.py
│   └── step4_evaluate.py
└── app/                               # Deployment package (Hugging Face / Docker)
    ├── .gitignore                     # Hugging Face Space cache ignore
    ├── README.md                      # Space settings & YAML frontmatter
    ├── Dockerfile                     # Streamlit container configuration
    ├── requirements.txt               # Lightweight CPU inference dependencies
    ├── app.py                         # Interactive Streamlit web interface
    └── model/
        └── efficientnet_b4_waste.pt   # Fine-tuned model checkpoint (~71 MB)
```

---

## 🚀 Reproducibility & Execution Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/waste-classifier.git
cd waste-classifier

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Pipeline Steps

```bash
# Step 0: Pre-flight sanity check
python src/sanity_check.py

# Step 1: Generate stratified splits & oversampling plan
python src/step1_split_data.py

# Step 2: Test dataloaders & preview augmentations
python src/step2_dataset.py

# Step 3: Train all three experimental configurations
# (Tip: add --smoke for a 2-minute dry run first)
python src/step3_train.py --mode all

# Step 4: Final statistical evaluation on the unseen test set
python src/step4_evaluate.py
```

---

## 🌐 Web Application & Deployment

The `app/` folder is designed to be pushed directly to **Hugging Face Spaces** (Docker SDK) or run locally.

### Local Execution
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

### Docker Execution
```bash
cd app
docker build -t waste-classifier-app .
docker run -p 8501:8501 waste-classifier-app
```
Access the application at `http://localhost:8501`.

---

## 📜 License & Citation

* **Code:** Licensed under the [MIT License](LICENSE).
* **Dataset:** Based on the Garbage Classification Dataset (v2) by Suman Kunwar. Please verify the upstream dataset license for commercial application.

