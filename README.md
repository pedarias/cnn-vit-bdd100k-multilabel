# CNN vs. ViT for Multi-Label Road Scene Classification on BDD100K

Code and artifacts for the paper:

> **Evaluation of Deep Learning Models for Classifying Weather Conditions on Roads and Highways**  
> Pedro H. A. Oliveira, Claiton de Oliveira, Silvio R. R. Sanches — UTFPR Cornélio Procópio  
> Submitted to *IEEE Access*

---

## Overview

Compares **MobileNetV2** (CNN) and **ViT-B/16** (Vision Transformer) for simultaneous
multi-label classification of weather, scene type, and time of day on the
[BDD100K](https://bdd-data.berkeley.edu/) dataset. In the main protocol both backbones are
fully frozen and only a shared 11-class sigmoid head is trained, isolating each
architecture's representational capacity.

Key results (frozen-backbone protocol, 13,097 test samples):

| Model | Global acc | Weather acc | Inconsistency | Latency (T4) |
|---|---|---|---|---|
| MobileNetV2 | 80.87% | 75.15% | 19.52% | 73 ms |
| ViT-B/16 | **83.84%** | **80.62%** | **14.04%** | 137 ms |

A controlled ablation on the same split also trains both models with the backbone
unfrozen. Fine-tuning wins in both cases (86.21% vs. 84.85% global), so freezing is a
training-efficiency optimum rather than an accuracy-maximizing choice; inference latency
and footprint are unchanged, being properties of the architecture rather than of the
training regime. Run `level_c_analysis.py` for the full ablation table.

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
└── level-c-out/                   # Per-sample predictions, latencies, training histories
    ├── predictions_{cnn,vit}_{frozen,finetune}.npz   (y_true, y_pred_bin, y_pred_proba)
    ├── predictions_ood_{cnn,vit}_{frozen,finetune}.npz
    ├── latencies_{cnn,vit}_{frozen,finetune}.npy     (50 measurements each)
    └── history_{cnn,vit}_{frozen,finetune}.{json,csv}  (per-epoch metrics)
```

All scripts read from `level-c-out/` using paths relative to their own location, so
they run from a fresh clone with no configuration. The published numbers come from the
`frozen` regime; set `REGIME = 'finetune'` at the top of a script to inspect the
fine-tuned models instead.

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
python level_c_analysis.py       # frozen-vs-fine-tune ablation, inconsistency, OOD foggy
python run_statistical_tests.py  # McNemar + Welch + Shapiro-Wilk, per-class weather P/R/F1
python level_b_analysis.py       # ECE calibration + error-detector recall
```

These reproduce the published statistics from the raw per-sample outputs, without
retraining: McNemar CNN vs ViT (subset accuracy) χ² = 323.46; Welch *t* = −10.43 with
Cohen's *d* = 2.09 on latency; McNemar on the inconsistency flag χ² = 185.19; ECE 1.58%
(CNN) vs 1.10% (ViT).

### 3. Figures

```bash
python gerar_pareto_e_benchmark.py   # pareto_frontier.png, benchmark_sota.png
python gerar_curvas_convergencia.py  # curvas_convergencia.png (from history_*.json)
```

`level_b_analysis.py` additionally writes `reliability_diagram.png`. Figure outputs are
git-ignored: the scripts plus `level-c-out/` are the sources of truth.

---

## Pre-trained checkpoints

Model checkpoints are not distributed with this repository. All reported metrics can be
reproduced from the per-sample artifacts in `level-c-out/` without them; retraining from
scratch takes roughly one Colab T4 session per configuration.

---

## Dataset

BDD100K is publicly available at <https://bdd-data.berkeley.edu/>.  
After curation (removing *foggy*, *parking lot*, *tunnel*, *gas station*, and *undefined*
labels), the working split is 87,292 images (61,110 train / 13,085 val / 13,097 test),
with 178 foggy images retained as a held-out OOD probe.

---

## Citation

```bibtex
@article{oliveira2026cnnvit,
  title   = {Evaluation of Deep Learning Models for Classifying Weather Conditions
             on Roads and Highways},
  author  = {Oliveira, Pedro H. A. and de Oliveira, Claiton and Sanches, Silvio R. R.},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

MIT
