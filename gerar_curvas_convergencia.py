"""
Gera as curvas de convergência (Loss, Accuracy, AUC) para a dissertação/artigo.
Lê os históricos por época exportados pelo notebook unificado (level_c_ablation_colab.ipynb)
em history_<arch>_<REGIME>.json, em vez de valores hard-coded.

Uso: python gerar_curvas_convergencia.py
Saída: curvas_convergencia.png / .pdf (300 DPI, formato publicação)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ============================================================
# Carregamento dos históricos (regime frozen = protocolo principal do artigo,
# 10 épocas; troque para 'finetune' para as curvas de 6 épocas)
# ============================================================
REGIME = 'frozen'
HIST_DIR = Path(__file__).resolve().parent / 'level-c-out'


def load_hist(arch):
    """Mapeia as keys do Keras para os nomes usados nos plots."""
    h = json.load(open(HIST_DIR / f'history_{arch}_{REGIME}.json'))
    return {
        'train_loss': h['loss'],          'val_loss': h['val_loss'],
        'train_acc':  h['binary_accuracy'], 'val_acc': h['val_binary_accuracy'],
        'train_auc':  h['auc'],           'val_auc': h['val_auc'],
    }


cnn = load_hist('cnn')
vit = load_hist('vit')
epochs = np.arange(1, len(cnn['train_loss']) + 1)

# ============================================================
# Configuração visual
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

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

# Cores consistentes
c_cnn_train = '#1f77b4'   # azul
c_cnn_val   = '#aec7e8'   # azul claro
c_vit_train = '#d62728'   # vermelho
c_vit_val   = '#ff9896'   # vermelho claro

lw_train = 2.0
lw_val = 2.0
ms = 5

# ============================================================
# Linha superior: CNN (MobileNetV2)
# ============================================================

# Loss
ax = axes[0, 0]
ax.plot(epochs, cnn['train_loss'], '-o', color=c_cnn_train, lw=lw_train, ms=ms, label='Treino')
ax.plot(epochs, cnn['val_loss'],   '-s', color=c_cnn_val,   lw=lw_val,   ms=ms, label='Validação')
ax.axhline(y=min(cnn['val_loss']), color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_ylabel('Binary Crossentropy')
ax.set_title('(a) Loss — MobileNetV2')
ax.legend(loc='upper right')
ax.set_ylim([0.15, 0.40])

# Accuracy
ax = axes[0, 1]
ax.plot(epochs, cnn['train_acc'], '-o', color=c_cnn_train, lw=lw_train, ms=ms, label='Treino')
ax.plot(epochs, cnn['val_acc'],   '-s', color=c_cnn_val,   lw=lw_val,   ms=ms, label='Validação')
ax.axhline(y=max(cnn['val_acc']), color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_title('(b) Binary Accuracy — MobileNetV2')
ax.legend(loc='lower right')
ax.set_ylim([0.84, 0.94])

# AUC
ax = axes[0, 2]
ax.plot(epochs, cnn['train_auc'], '-o', color=c_cnn_train, lw=lw_train, ms=ms, label='Treino')
ax.plot(epochs, cnn['val_auc'],   '-s', color=c_cnn_val,   lw=lw_val,   ms=ms, label='Validação')
ax.axhline(y=max(cnn['val_auc']), color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_title('(c) AUC — MobileNetV2')
ax.legend(loc='lower right')
ax.set_ylim([0.84, 0.96])

# ============================================================
# Linha inferior: ViT-B/16
# ============================================================

# Loss
ax = axes[1, 0]
ax.plot(epochs, vit['train_loss'], '-o', color=c_vit_train, lw=lw_train, ms=ms, label='Treino')
ax.plot(epochs, vit['val_loss'],   '-s', color=c_vit_val,   lw=lw_val,   ms=ms, label='Validação')
ax.axhline(y=min(vit['val_loss']), color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_ylabel('Binary Crossentropy')
ax.set_xlabel('Época')
ax.set_title('(d) Loss — ViT-B/16')
ax.legend(loc='upper right')
ax.set_ylim([0.15, 0.40])

# Accuracy
ax = axes[1, 1]
ax.plot(epochs, vit['train_acc'], '-o', color=c_vit_train, lw=lw_train, ms=ms, label='Treino')
ax.plot(epochs, vit['val_acc'],   '-s', color=c_vit_val,   lw=lw_val,   ms=ms, label='Validação')
ax.axhline(y=max(vit['val_acc']), color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('Época')
ax.set_title('(e) Binary Accuracy — ViT-B/16')
ax.legend(loc='lower right')
ax.set_ylim([0.88, 0.94])

# AUC
ax = axes[1, 2]
ax.plot(epochs, vit['train_auc'], '-o', color=c_vit_train, lw=lw_train, ms=ms, label='Treino')
ax.plot(epochs, vit['val_auc'],   '-s', color=c_vit_val,   lw=lw_val,   ms=ms, label='Validação')
ax.axhline(y=max(vit['val_auc']), color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('Época')
ax.set_title('(f) AUC — ViT-B/16')
ax.legend(loc='lower right')
ax.set_ylim([0.84, 0.96])

# ============================================================
# Ajustes globais
# ============================================================

for ax in axes.flat:
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_xlim([0.5, 10.5])

plt.tight_layout(h_pad=2.5, w_pad=1.5)
plt.savefig('curvas_convergencia.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('curvas_convergencia.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Figuras salvas: curvas_convergencia.png / .pdf")
plt.show()
