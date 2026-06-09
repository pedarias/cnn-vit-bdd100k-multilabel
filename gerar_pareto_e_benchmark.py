"""
Gera dois gráficos para o artigo IEEE Access:
  1. pareto_frontier.png/pdf — Pareto frontier of efficiency-accuracy trade-offs
                               (latency vs accuracy + VRAM vs accuracy)
  2. benchmark_sota.png/pdf  — Trainable params vs accuracy comparing our
                               protocol against published BDD100K baselines

Uso: python gerar_pareto_e_benchmark.py
Estilo consistente com gerar_curvas_convergencia.py (300 DPI, fonte serif).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ============================================================
# Visual configuration (matches existing convergence figure)
# ============================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
})

C_CNN = '#1f77b4'   # blue   — our MobileNetV2
C_VIT = '#d62728'   # red    — our ViT-B/16
C_LIT = '#7f7f7f'   # grey   — literature baselines

# ============================================================
# Figure 1: Pareto frontier
# ============================================================

# Canonical numbers (frozen + fine-tune), test set 13,097
# acc = global; foot = FP32 weight footprint (MB); par = total params (M)
cnn_fr = dict(acc=80.87, foot=13.7, par=3.59, lat=73.12)
cnn_ft = dict(acc=84.85, foot=13.7, par=3.59, lat=73.12)   # mesma arquitetura → mesmo footprint/lat
vit_fr = dict(acc=83.84, foot=332.6, par=87.19, lat=136.89)
vit_ft = dict(acc=86.21, foot=332.6, par=87.19, lat=136.89)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# (a) Latency vs Accuracy — 4 pontos (frozen vazado, fine-tune preenchido)
ax = axes[0]
for d, c, mk, lab in [(cnn_fr, C_CNN, 'o', 'MobileNetV2 (frozen)'),
                      (cnn_ft, C_CNN, 'o', 'MobileNetV2 (fine-tune)'),
                      (vit_fr, C_VIT, 's', 'ViT-B/16 (frozen)'),
                      (vit_ft, C_VIT, 's', 'ViT-B/16 (fine-tune)')]:
    fill = c if 'fine-tune' in lab else 'white'
    ax.scatter(d['lat'], d['acc'], s=170, color=fill, edgecolor=c,
               marker=mk, linewidths=1.8, zorder=5, label=lab)
# seta frozen→finetune (mesma arquitetura, ganho de acurácia, latência igual)
for fr, ft, c in [(cnn_fr, cnn_ft, C_CNN), (vit_fr, vit_ft, C_VIT)]:
    ax.annotate('', xy=(ft['lat'], ft['acc']), xytext=(fr['lat'], fr['acc']),
                arrowprops=dict(arrowstyle='->', color=c, lw=1.2, alpha=0.7))
ax.set_xlabel('Inference Latency (ms)')
ax.set_ylabel('Global Accuracy (\\%)')
ax.set_title('(a) Latency vs. Accuracy')
ax.set_xlim([60, 150]); ax.set_ylim([79, 87.5])
ax.grid(True, alpha=0.3, ls='--')
ax.legend(loc='lower right', framealpha=0.9, fontsize=8)

# (b) Parameter footprint vs Accuracy (log scale on x)
ax = axes[1]
for d, c, mk, lab in [(cnn_fr, C_CNN, 'o', 'MobileNetV2 (frozen)'),
                      (cnn_ft, C_CNN, 'o', 'MobileNetV2 (fine-tune)'),
                      (vit_fr, C_VIT, 's', 'ViT-B/16 (frozen)'),
                      (vit_ft, C_VIT, 's', 'ViT-B/16 (fine-tune)')]:
    fill = c if 'fine-tune' in lab else 'white'
    ax.scatter(d['foot'], d['acc'], s=170, color=fill, edgecolor=c,
               marker=mk, linewidths=1.8, zorder=5, label=lab)
for fr, ft, c in [(cnn_fr, cnn_ft, C_CNN), (vit_fr, vit_ft, C_VIT)]:
    ax.annotate('', xy=(ft['foot'], ft['acc']), xytext=(fr['foot'], fr['acc']),
                arrowprops=dict(arrowstyle='->', color=c, lw=1.2, alpha=0.7))
ax.annotate('', xy=(vit_fr['foot'], 80.1), xytext=(cnn_fr['foot'], 80.1),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
ax.text(np.sqrt(cnn_fr['foot'] * vit_fr['foot']), 79.55,
        r'$\sim$24$\times$', ha='center', fontsize=9, style='italic')
ax.set_xscale('log')
ax.set_xlabel('Model Weight Footprint (MB, FP32, log scale)')
ax.set_ylabel('Global Accuracy (\\%)')
ax.set_title('(b) Memory Footprint vs. Accuracy')
ax.set_xlim([8, 600]); ax.set_ylim([79, 87.5])
ax.grid(True, alpha=0.3, ls='--', which='both')
ax.legend(loc='lower right', framealpha=0.9, fontsize=8)

plt.tight_layout()
plt.savefig('pareto_frontier.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('pareto_frontier.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
print('Saved: pareto_frontier.png / .pdf')

# ============================================================
# Figure 2: Trainable params vs Accuracy (vs literature)
# ============================================================

# Literature baselines: (trainable_params_M, GLOBAL_accuracy_%, label)
# Prykhodchenko global = mean of the 3 task accuracies (not weather-only)
lit_points = [
    (15.8, 55.0,  'Yu et al.\\ (DLA-34)'),
    (44.5, 81.0,  'BDD100K Zoo (ResNet-101)'),
    (11.7, 80.84, 'Prykhodchenko (ResNet-18 FT)'),
    (86.5, 80.18, 'Prykhodchenko (ViT-B/16 FT)'),
]

# Our points (frozen protocol, canonical run, global accuracy)
ours_cnn = (1.33, 80.87, 'MobileNetV2 (Ours)')
ours_vit = (0.80, 83.84, 'ViT-B/16 (Ours)')

fig, ax = plt.subplots(figsize=(8, 5.5))

# Shaded Pareto-optimal region (upper-left quadrant)
ax.axvspan(0.5, 3.0, ymin=0.55, ymax=1.0,
           alpha=0.10, color='green', zorder=0)
ax.text(1.0, 88.8, 'Pareto-optimal region',
        fontsize=9, style='italic', color='darkgreen', alpha=0.85)

# Literature points
for x, y, lab in lit_points:
    ax.scatter(x, y, s=130, color=C_LIT, edgecolor='black',
               marker='o', alpha=0.85, zorder=4)
    ax.annotate(lab, xy=(x, y), xytext=(8, -3),
                textcoords='offset points', fontsize=8.5, color='#333333')

# Our points (highlighted)
ax.scatter(ours_cnn[0], ours_cnn[1], s=220, color=C_CNN, edgecolor='black',
           marker='*', zorder=6, label=ours_cnn[2])
ax.annotate(ours_cnn[2], xy=ours_cnn[:2], xytext=(10, -12),
            textcoords='offset points', fontsize=9, fontweight='bold', color=C_CNN)

ax.scatter(ours_vit[0], ours_vit[1], s=220, color=C_VIT, edgecolor='black',
           marker='*', zorder=6, label=ours_vit[2])
ax.annotate(ours_vit[2], xy=ours_vit[:2], xytext=(10, 5),
            textcoords='offset points', fontsize=9, fontweight='bold', color=C_VIT)

# Arrow from Prykhodchenko ViT-FT to ours (global vs global)
ax.annotate(
    '',
    xy=(ours_vit[0], ours_vit[1]),
    xytext=(86.5, 80.18),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.3,
                    connectionstyle='arc3,rad=-0.25'),
)
ax.text(8.0, 78.8,
        'Frozen-backbone protocol:\n$-108\\times$ params, $+3.66$ p.p.',
        fontsize=8.5, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9))

ax.set_xscale('log')
ax.set_xlabel('Trainable Parameters (M, log scale)')
ax.set_ylabel('Global Accuracy (\\%)')
ax.set_title('Frozen Head vs.\\ End-to-End Training on BDD100K Tagging')
ax.set_xlim([0.5, 200])
ax.set_ylim([50, 90])
ax.grid(True, alpha=0.3, ls='--', which='both')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))

plt.tight_layout()
plt.savefig('benchmark_sota.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('benchmark_sota.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
print('Saved: benchmark_sota.png / .pdf')

print('\nDone.')
