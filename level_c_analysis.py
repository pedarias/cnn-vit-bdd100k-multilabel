"""
NIVEL C - analise local dos arrays exportados pela ablacao no Colab.
Roda APOS baixar predictions_*.npz / latencies_*.npy / predictions_ood_*.npz
de /content/level_c_out para o diretorio level-c-out/ deste repositorio.

Uso: python level_c_analysis.py

Produz:
  - tabela frozen vs fine-tune (global + por tarefa) para CNN e ViT
  - McNemar pareado frozen-vs-finetune (mesmo test set)  -> evidencia causal
  - inconsistencia logica por condicao
  - degradacao OOD (foggy): scene/time acc + taxa de inconsistencia
"""
import numpy as np
from pathlib import Path
from scipy import stats
try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAVE_SM = True
except Exception:
    HAVE_SM = False

DIR = Path(__file__).resolve().parent / 'level-c-out'

CLASS_NAMES = [
    'scene_city street', 'scene_highway', 'scene_residential',
    'time_dawn/dusk', 'time_daytime', 'time_night',
    'weather_clear', 'weather_overcast', 'weather_partly cloudy',
    'weather_rainy', 'weather_snowy',
]
scene_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith('scene_')]
time_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith('time_')]
weather_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith('weather_')]


def load(cond):
    p = DIR / f'predictions_{cond}.npz'
    if not p.exists():
        return None
    d = np.load(p)
    return d['y_true'].astype(int), d['y_pred_bin'].astype(int), d['y_pred_proba']


def task_argmax_acc(y_true, proba, idx):
    return (y_true[:, idx].argmax(1) == proba[:, idx].argmax(1)).mean()


def global_acc(y_true, proba):
    return np.mean([task_argmax_acc(y_true, proba, ix)
                    for ix in (scene_idx, time_idx, weather_idx)])


def subset_correct(y_true, y_pred_bin):
    return (y_true == y_pred_bin).all(1)


def inconsistency(y_pred_bin):
    t = y_pred_bin[:, time_idx].sum(1)
    w = y_pred_bin[:, weather_idx].sum(1)
    return ((t != 1) | (w != 1)).mean()


print('=' * 64)
print('C1. ABLACAO FROZEN vs FINE-TUNE (mesmo split 87K, mesmo head)')
print('=' * 64)
for arch in ['cnn', 'vit']:
    fr = load(f'{arch}_frozen')
    ft = load(f'{arch}_finetune')
    if fr is None or ft is None:
        print(f'\n[{arch}] faltam arrays (frozen={fr is not None}, finetune={ft is not None})')
        continue
    yt, frb, frp = fr
    _, ftb, ftp = ft
    print(f'\n--- {arch.upper()} ---')
    for label, (b, p) in [('frozen', (frb, frp)), ('finetune', (ftb, ftp))]:
        print(f'  {label:9s}: global={global_acc(yt, p):.4f}  '
              f'scene={task_argmax_acc(yt, p, scene_idx):.4f}  '
              f'time={task_argmax_acc(yt, p, time_idx):.4f}  '
              f'weather={task_argmax_acc(yt, p, weather_idx):.4f}  '
              f'inc={inconsistency(b)*100:.2f}%')
    # McNemar pareado: frozen vs finetune no subset-accuracy (11 labels)
    c_fr = subset_correct(yt, frb)
    c_ft = subset_correct(yt, ftb)
    n01 = int((~c_fr & c_ft).sum())   # frozen erra, finetune acerta
    n10 = int((c_fr & ~c_ft).sum())   # frozen acerta, finetune erra
    print(f'  McNemar frozen-vs-finetune: finetune-only={n01}  frozen-only={n10}')
    if HAVE_SM and (n01 + n10) > 0:
        res = mcnemar([[0, n01], [n10, 0]], exact=False, correction=True)
        print(f'    chi2={res.statistic:.2f}  p={res.pvalue:.3g}  '
              f'-> {"frozen melhor" if n10>n01 else "finetune melhor"} '
              f'({"significativo" if res.pvalue<0.05 else "n.s."})')


print()
print('=' * 64)
print('C2. DEGRADACAO OOD (imagens foggy descartadas na curadoria)')
print('=' * 64)
print('NB: weather correto e impossivel (foggy nao esta nas 5 classes).')
print('    Medimos: (a) scene/time acc (essas labels SAO validas), (b)')
print('    inconsistencia logica, (c) quanto o modelo "alucina" weather.')
for cond in ['cnn_frozen', 'cnn_finetune', 'vit_frozen', 'vit_finetune']:
    p = DIR / f'predictions_ood_{cond}.npz'
    if not p.exists():
        continue
    d = np.load(p)
    yt, b, pr = d['y_true'].astype(int), d['y_pred_bin'].astype(int), d['y_pred_proba']
    sc = task_argmax_acc(yt, pr, scene_idx)
    tm = task_argmax_acc(yt, pr, time_idx)
    inc = inconsistency(b)
    # confianca media na classe weather mais ativada (alucinacao)
    w_conf = pr[:, weather_idx].max(1).mean()
    print(f'\n  {cond:13s}: scene={sc:.4f}  time={tm:.4f}  inc={inc*100:.2f}%  '
          f'weather_conf_max(media)={w_conf:.3f}')

print()
print('Interpretacao esperada: se o ViT degrada MENOS que a CNN em foggy')
print('(maior scene/time acc, menor inconsistencia), isso confirma robustez')
print('OOD superior -> transforma o vies de curadoria em contribuicao.')
