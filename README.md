# CNN vs. ViT for Multi-Label Road Scene Classification on BDD100K

Code and artifacts for the paper:

> **Evaluation of Deep Learning Models for Classifying Weather Conditions on Roads and Highways**  
> Pedro H. A. Oliveira, Claiton de Oliveira, Silvio R. R. Sanches — UTFPR Cornélio Procópio  
> *IEEE Access* (under review)

---

## Overview

Compares **MobileNetV2** (CNN) and **ViT-B/16** (Vision Transformer) for simultaneous
multi-label classification of weather, scene type, and time of day on the
[BDD100K](https://bdd-data.berkeley.edu/) dataset. Both backbones are fully frozen; only
a shared 11-class sigmoid head is trained, isolating each architecture's representational
capacity.

Key results (frozen-backbone protocol, 13,097 test samples):

| Model | Global acc | Weather acc | Inconsistency | Latency (T4) |
|---|---|---|---|---|
| MobileNetV2 | 80.87% | 75.15% | 19.52% | 73 ms |
| ViT-B/16 | **83.84%** | **80.62%** | **14.04%** | 137 ms |

---

## Repository structure

```
.
├── level_c_ablation_colab.ipynb   # Main notebook: trains all 4 configs, exports artifacts
├── level_c_analysis.py            # Per-config accuracy, inconsistency, OOD foggy
├── run_statistical_tests.py       # McNemar (CNN vs ViT) + Welch t-test (latency)
├── level_b_analysis.py            # ECE calibration + error-detector recall
├── gerar_pareto_e_benchmark.py    # Pareto frontier + SOTA benchmark figures
├── gerar_curvas_convergencia.py   # Training convergence curves
└── level-c-out/                   # Per-sample predictions and latency arrays
    ├── predictions_{cnn,vit}_{frozen,finetune}.npz   (y_true, y_pred_bin, y_pred_proba)
    ├── predictions_ood_{cnn,vit}_{frozen,finetune}.npz
    └── latencies_{cnn,vit}_{frozen,finetune}.npy
```

---

## Reproducing the results

### 1. Training (Google Colab, GPU T4)

Open `level_c_ablation_colab.ipynb` in Colab. Before running:

- Mount your Google Drive and set `DRIVE_BASE` to the folder containing the BDD100K
  images and labels (the notebook expects the standard BDD100K directory structure).
- The notebook trains all four configurations (MobileNetV2/ViT-B/16 × frozen/fine-tune),
  evaluates on the test set, runs the OOD foggy test, and exports all artifacts to
  `level-c-out/`.

### 2. Statistical analysis (local)

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the analysis scripts against the exported artifacts:

```bash
python level_c_analysis.py       # accuracy, inconsistency, per-class, OOD
python run_statistical_tests.py  # McNemar + Welch + Shapiro-Wilk
python level_b_analysis.py       # ECE + error-detector recall
```

### 3. Figures

```bash
python gerar_pareto_e_benchmark.py   # pareto_frontier.png, benchmark_sota.png
python gerar_curvas_convergencia.py  # curvas_convergencia.png
```

---

## Pre-trained checkpoints

The four model checkpoints (`cnn_frozen_best.keras`, `cnn_finetune_best.keras`,
`vit_frozen_best.keras`, `vit_finetune_best.keras`) are available as assets in the
[v1.0 Release](../../releases/tag/v1.0).

---

## Dataset

BDD100K is publicly available at <https://bdd-data.berkeley.edu/>.  
After curation (removing *foggy*, *parking lot*, *tunnel*, *gas station*, and *undefined*
labels), the working split is 87,292 images (61,110 train / 13,085 val / 13,097 test),
with 178 foggy images retained as a held-out OOD probe.

---

## Citation

```bibtex
@article{oliveira2025cnnvit,
  title   = {Evaluation of Deep Learning Models for Classifying Weather Conditions
             on Roads and Highways},
  author  = {Oliveira, Pedro H. A. and de Oliveira, Claiton and Sanches, Silvio R. R.},
  journal = {IEEE Access},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT
