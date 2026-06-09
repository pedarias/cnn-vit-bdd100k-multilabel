"""
Roda os testes estatisticos formais (McNemar para acuracia, Welch t-test para latencia)
sobre os dados exportados dos notebooks re-executados no Colab.

Saida: imprime relatorio com os valores a substituir nos placeholders do access.tex
       e tambem salva metricas atualizadas em statistical_results.txt
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_fscore_support,
)

CNN_DIR = Path('/home/polivei/Documents/diss/notebooks/cnn-contexts')
VIT_DIR = Path('/home/polivei/Documents/diss/notebooks/vit-contents')

# ============================================================
# 1. Load data
# ============================================================
lat_cnn = np.load(CNN_DIR / 'latencies_cnn.npy')
lat_vit = np.load(VIT_DIR / 'latencies_vit.npy')

pcnn = np.load(CNN_DIR / 'predictions_cnn.npz')
pvit = np.load(VIT_DIR / 'predictions_vit.npz')

y_true = pcnn['y_true'].astype(int)
y_cnn = pcnn['y_pred_bin'].astype(int)
y_vit = pvit['y_pred_bin'].astype(int)
proba_cnn = pcnn['y_pred_proba']
proba_vit = pvit['y_pred_proba']
N, C = y_true.shape

print('=' * 70)
print('1. DATASET METRICS (updated from re-run)')
print('=' * 70)
print(f'Test samples: {N}')
print(f'Classes: {C}')

# Class names: scene_*, time_*, weather_*  (ordem alfabetica de MultiLabelBinarizer)
# Sera reconstruida abaixo a partir do CSV de distribuicao se necessario
print()
print('=' * 70)
print('2. LATENCY (50 measurements each)')
print('=' * 70)
print(f'CNN: mean={lat_cnn.mean():.2f} ms, std={lat_cnn.std(ddof=1):.2f} ms, '
      f'min={lat_cnn.min():.2f}, max={lat_cnn.max():.2f}')
print(f'ViT: mean={lat_vit.mean():.2f} ms, std={lat_vit.std(ddof=1):.2f} ms, '
      f'min={lat_vit.min():.2f}, max={lat_vit.max():.2f}')
print(f'FPS CNN: {1000/lat_cnn.mean():.2f}')
print(f'FPS ViT: {1000/lat_vit.mean():.2f}')

# Shapiro-Wilk for normality
sw_cnn = stats.shapiro(lat_cnn)
sw_vit = stats.shapiro(lat_vit)
print(f'\nShapiro-Wilk CNN: W={sw_cnn.statistic:.4f}, p={sw_cnn.pvalue:.4g}')
print(f'Shapiro-Wilk ViT: W={sw_vit.statistic:.4f}, p={sw_vit.pvalue:.4g}')

# Welch t-test
t_stat, t_p = stats.ttest_ind(lat_cnn, lat_vit, equal_var=False)
# Cohen's d (with pooled sd, since means are very different)
pooled_sd = np.sqrt((lat_cnn.var(ddof=1) + lat_vit.var(ddof=1)) / 2)
cohens_d = abs(lat_cnn.mean() - lat_vit.mean()) / pooled_sd

print(f"\nWelch t-test (CNN vs ViT latency):")
print(f"  t = {t_stat:.4f}")
print(f"  p = {t_p:.4g}")
print(f"  Cohen's d = {cohens_d:.4f}")

# ============================================================
# 3. Per-task accuracy (Scene / Time of Day / Weather)
# ============================================================
# Reconstruct class names: alphabetical order of MultiLabelBinarizer.
# scene: city street, highway, residential
# time: dawn/dusk, daytime, night
# weather: clear, overcast, partly cloudy, rainy, snowy
CLASS_NAMES = [
    'scene_city street', 'scene_highway', 'scene_residential',
    'time_dawn/dusk', 'time_daytime', 'time_night',
    'weather_clear', 'weather_overcast', 'weather_partly cloudy',
    'weather_rainy', 'weather_snowy',
]
assert len(CLASS_NAMES) == C

scene_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith('scene_')]
time_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith('time_')]
weather_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith('weather_')]

# ============================================================
# Per-class precision/recall/F1 (Weather only — most cited in paper)
# ============================================================
print()
print('=' * 70)
print('2.5 PER-CLASS WEATHER METRICS (P / R / F1)')
print('=' * 70)
def report_task_per_class(task_label, idxs, prefix):
    names = [CLASS_NAMES[i].replace(prefix, '') for i in idxs]
    print(f'\n=== {task_label} ===')
    yt_arg = y_true[:, idxs].argmax(axis=1)
    for model_name, proba in [('CNN', proba_cnn), ('ViT', proba_vit)]:
        yp_arg = proba[:, idxs].argmax(axis=1)
        p, r, f, _ = precision_recall_fscore_support(
            yt_arg, yp_arg, labels=list(range(len(idxs))),
            average=None, zero_division=0,
        )
        print(f'  -- {model_name} (argmax single-label) --')
        for name, pp, rr, ff in zip(names, p, r, f):
            print(f'    {name:18s}  P={pp:.4f}  R={rr:.4f}  F1={ff:.4f}')

report_task_per_class('Scene', scene_idx, 'scene_')
report_task_per_class('Time of Day', time_idx, 'time_')
report_task_per_class('Weather', weather_idx, 'weather_')

# Micro-averaged AUC-ROC
auc_cnn = roc_auc_score(y_true, proba_cnn, average='micro')
auc_vit = roc_auc_score(y_true, proba_vit, average='micro')
print(f'\nAUC-ROC (micro): CNN={auc_cnn:.4f}  ViT={auc_vit:.4f}')

# ============================================================
# Logical inconsistency rate
# (sample is "inconsistent" if more than one mutually-exclusive class
#  is active simultaneously within a task — taking threshold 0.5)
# ============================================================
def inconsistency_notebook(y_pred_bin):
    """Original notebook definition (cell 27 of CNN nb): a sample is inconsistent
    if (a) >1 time labels active, (b) >1 weather labels active,
    (c) 0 time labels active, or (d) 0 weather labels active.
    Scene is NOT checked."""
    t = y_pred_bin[:, time_idx].sum(axis=1)
    w = y_pred_bin[:, weather_idx].sum(axis=1)
    return (t > 1) | (w > 1) | (t == 0) | (w == 0)

ic = inconsistency_notebook(y_cnn)
iv = inconsistency_notebook(y_vit)
print(f'\nLogical inconsistency (notebook definition):')
print(f'  CNN: {ic.sum()}/{N} = {ic.mean()*100:.2f}%')
print(f'  ViT: {iv.sum()}/{N} = {iv.mean()*100:.2f}%')
print(f'  Absolute reduction: {(ic.mean()-iv.mean())*100:+.2f} p.p.')
print(f'  Relative reduction: {(1 - iv.mean()/ic.mean())*100:.1f}%')

def task_accuracy(y_true_t, y_pred_t, idxs):
    """Subset accuracy on the multi-label sub-vector for the given task."""
    yt = y_true_t[:, idxs]
    yp = y_pred_t[:, idxs]
    # A sample is correct if the entire sub-vector matches
    return (yt == yp).all(axis=1).mean()

def task_binary_accuracy(y_true_t, y_pred_t, idxs):
    """Element-wise binary accuracy across the task sub-vector (matches Keras
    binary_accuracy averaged within task — CLAUDE.md convention)."""
    yt = y_true_t[:, idxs]
    yp = y_pred_t[:, idxs]
    return (yt == yp).mean()

def task_argmax_accuracy(y_true_t, proba_t, idxs):
    """Single-label accuracy via argmax within the task sub-vector.
    Matches the original notebooks' per-task accuracy reported in CLAUDE.md."""
    yt = y_true_t[:, idxs].argmax(axis=1)
    yp = proba_t[:, idxs].argmax(axis=1)
    return (yt == yp).mean()

print()
print('=' * 70)
print('3. PER-TASK ACCURACY')
print('=' * 70)
print('-- Subset accuracy per task (all labels in sub-vector match) --')
for label, idxs in [('Scene', scene_idx), ('Time of Day', time_idx), ('Weather', weather_idx)]:
    acc_cnn = task_accuracy(y_true, y_cnn, idxs)
    acc_vit = task_accuracy(y_true, y_vit, idxs)
    print(f'{label}: CNN={acc_cnn*100:.2f}%  ViT={acc_vit*100:.2f}%  diff={(acc_vit-acc_cnn)*100:+.2f} p.p.')

print('\n-- Argmax (single-label) accuracy per task — matches CLAUDE.md --')
for label, idxs in [('Scene', scene_idx), ('Time of Day', time_idx), ('Weather', weather_idx)]:
    aacc_cnn = task_argmax_accuracy(y_true, proba_cnn, idxs)
    aacc_vit = task_argmax_accuracy(y_true, proba_vit, idxs)
    print(f'{label}: CNN={aacc_cnn*100:.2f}%  ViT={aacc_vit*100:.2f}%  diff={(aacc_vit-aacc_cnn)*100:+.2f} p.p.')

aacc_g_cnn = (
    task_argmax_accuracy(y_true, proba_cnn, scene_idx)
    + task_argmax_accuracy(y_true, proba_cnn, time_idx)
    + task_argmax_accuracy(y_true, proba_cnn, weather_idx)
) / 3
aacc_g_vit = (
    task_argmax_accuracy(y_true, proba_vit, scene_idx)
    + task_argmax_accuracy(y_true, proba_vit, time_idx)
    + task_argmax_accuracy(y_true, proba_vit, weather_idx)
) / 3
print(f'Global mean argmax: CNN={aacc_g_cnn*100:.2f}%  ViT={aacc_g_vit*100:.2f}%  diff={(aacc_g_vit-aacc_g_cnn)*100:+.2f} p.p.')

print('\n-- Binary accuracy per task (Keras binary_accuracy convention) --')
for label, idxs in [('Scene', scene_idx), ('Time of Day', time_idx), ('Weather', weather_idx)]:
    bacc_cnn = task_binary_accuracy(y_true, y_cnn, idxs)
    bacc_vit = task_binary_accuracy(y_true, y_vit, idxs)
    print(f'{label}: CNN={bacc_cnn*100:.2f}%  ViT={bacc_vit*100:.2f}%  diff={(bacc_vit-bacc_cnn)*100:+.2f} p.p.')

bacc_g_cnn = (y_true == y_cnn).mean()
bacc_g_vit = (y_true == y_vit).mean()
print(f'Global binary accuracy: CNN={bacc_g_cnn*100:.2f}%  ViT={bacc_g_vit*100:.2f}%  diff={(bacc_g_vit-bacc_g_cnn)*100:+.2f} p.p.')

# Global mean accuracy
acc_g_cnn = (task_accuracy(y_true, y_cnn, scene_idx)
             + task_accuracy(y_true, y_cnn, time_idx)
             + task_accuracy(y_true, y_cnn, weather_idx)) / 3
acc_g_vit = (task_accuracy(y_true, y_vit, scene_idx)
             + task_accuracy(y_true, y_vit, time_idx)
             + task_accuracy(y_true, y_vit, weather_idx)) / 3
print(f'Global mean: CNN={acc_g_cnn*100:.2f}%  ViT={acc_g_vit*100:.2f}%  diff={(acc_g_vit-acc_g_cnn)*100:+.2f} p.p.')

# ============================================================
# 4. McNemar test (paired classifier comparison)
# ============================================================
print()
print('=' * 70)
print('4. McNEMAR TEST (paired comparison on 13,115 test samples)')
print('=' * 70)

# Define "correct" per sample: all 11 labels match (subset accuracy)
correct_cnn = (y_true == y_cnn).all(axis=1)
correct_vit = (y_true == y_vit).all(axis=1)

# Contingency table:
#                  ViT correct | ViT wrong
# CNN correct          a              b
# CNN wrong            c              d
a = ((correct_cnn) & (correct_vit)).sum()
b = ((correct_cnn) & (~correct_vit)).sum()
c = ((~correct_cnn) & (correct_vit)).sum()
d = ((~correct_cnn) & (~correct_vit)).sum()
print(f'Subset-correct contingency table (full 11-label match):')
print(f'                    ViT correct   ViT wrong')
print(f'  CNN correct:        {a:>8d}    {b:>8d}')
print(f'  CNN wrong:          {c:>8d}    {d:>8d}')

table = [[a, b], [c, d]]
mc = mcnemar(table, exact=False, correction=True)
print(f'McNemar (subset-correct): chi2={mc.statistic:.4f}, p={mc.pvalue:.4g}')

# Also compute per-task McNemar (more interpretable)
print('\nPer-task McNemar (only flips on the relevant sub-vector):')
for label, idxs in [('Scene', scene_idx), ('Time of Day', time_idx), ('Weather', weather_idx)]:
    cc = (y_true[:, idxs] == y_cnn[:, idxs]).all(axis=1)
    cv = (y_true[:, idxs] == y_vit[:, idxs]).all(axis=1)
    b_ = (cc & (~cv)).sum()
    c_ = ((~cc) & cv).sum()
    tab = [[(cc & cv).sum(), b_], [c_, ((~cc) & (~cv)).sum()]]
    mc_ = mcnemar(tab, exact=False, correction=True)
    print(f'  {label}: chi2={mc_.statistic:.4f}, p={mc_.pvalue:.4g}, '
          f'CNN-only correct={b_}, ViT-only correct={c_}')
