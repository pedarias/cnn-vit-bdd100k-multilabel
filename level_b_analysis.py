"""
Nivel B - analises sobre os dados JA exportados (sem re-treino).
B1: validar inconsistencia vs ground truth (P(erro|inconsistente) vs P(erro|consistente))
B2: variante da inconsistencia com e sem o grupo scene (robustez da exclusao)
B3: calibracao (ECE + reliability diagram)
B4: estatistica robusta de latencia (mediana, IQR, p95/p99, IC bootstrap)

Uso: conda run -n diss python notebooks/level_b_analysis.py
Saida: imprime relatorio + salva reliability_diagram.png em notebooks/
"""
import numpy as np
from pathlib import Path

CNN_DIR = Path('/home/polivei/Documents/diss/notebooks/cnn-contexts')
VIT_DIR = Path('/home/polivei/Documents/diss/notebooks/vit-contents')
OUT_DIR = Path('/home/polivei/Documents/diss/notebooks')

pcnn = np.load(CNN_DIR / 'predictions_cnn.npz')
pvit = np.load(VIT_DIR / 'predictions_vit.npz')
y_true = pcnn['y_true'].astype(int)
y_cnn = pcnn['y_pred_bin'].astype(int)
y_vit = pvit['y_pred_bin'].astype(int)
proba_cnn = pcnn['y_pred_proba']
proba_vit = pvit['y_pred_proba']
lat_cnn = np.load(CNN_DIR / 'latencies_cnn.npy')
lat_vit = np.load(VIT_DIR / 'latencies_vit.npy')
N, C = y_true.shape

CLASS_NAMES = [
    'scene_city street', 'scene_highway', 'scene_residential',
    'time_dawn/dusk', 'time_daytime', 'time_night',
    'weather_clear', 'weather_overcast', 'weather_partly cloudy',
    'weather_rainy', 'weather_snowy',
]
scene_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith('scene_')]
time_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith('time_')]
weather_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith('weather_')]


def inconsistent_tw(yp):
    """Definicao do artigo: so time+weather; !=1 ativo em algum grupo."""
    t = yp[:, time_idx].sum(axis=1)
    w = yp[:, weather_idx].sum(axis=1)
    return (t != 1) | (w != 1)


def inconsistent_tws(yp):
    """Variante incluindo scene."""
    t = yp[:, time_idx].sum(axis=1)
    w = yp[:, weather_idx].sum(axis=1)
    s = yp[:, scene_idx].sum(axis=1)
    return (t != 1) | (w != 1) | (s != 1)


def subset_wrong(yp):
    """True onde a predicao multi-rotulo (11 labels) NAO bate exatamente."""
    return ~(y_true == yp).all(axis=1)


print('=' * 70)
print('B1. INCONSISTENCIA COMO DETECTOR DE ERRO SEM GROUND TRUTH')
print('=' * 70)
print('NB: P(erro|inconsistente)=100% e tautologico (>1 ou 0 labels nunca bate o')
print('    subset-acc, que exige exatamente 1 por grupo). O enquadramento correto e')
print('    inconsistencia como flag de erro em tempo de inferencia (sem rotulo):')
print('    precisao=fracao de flags que sao erro; recall=fracao dos erros flagrada.')
for name, yp in [('CNN', y_cnn), ('ViT', y_vit)]:
    inc = inconsistent_tw(yp)
    wrong = subset_wrong(yp)
    tp = (inc & wrong).sum()       # flag e erro
    fp = (inc & ~wrong).sum()      # flag mas correto
    prec = tp / inc.sum()          # precisao do flag
    rec = tp / wrong.sum()         # recall de erro
    base = wrong.mean()            # taxa de erro global (baseline)
    print(f'\n{name}:')
    print(f'  inconsistentes (flags): {inc.sum()}/{N} = {inc.mean()*100:.2f}%')
    print(f'  taxa de subset-erro global (baseline): {base*100:.2f}%')
    print(f'  precisao do flag = {prec*100:.2f}%  (lift vs baseline = {prec/base:.2f}x)')
    print(f'  recall de erro   = {rec*100:.2f}%  '
          f'(o flag captura essa fracao dos erros, sem usar ground truth)')


print()
print('=' * 70)
print('B2. INCONSISTENCIA: COM vs SEM SCENE (robustez da exclusao)')
print('=' * 70)
for label, fn in [('time+weather (artigo)', inconsistent_tw),
                  ('time+weather+scene', inconsistent_tws)]:
    ic, iv = fn(y_cnn).mean(), fn(y_vit).mean()
    print(f'\n{label}:')
    print(f'  CNN={ic*100:.2f}%  ViT={iv*100:.2f}%  '
          f'abs={-(ic-iv)*100:+.2f} p.p.  rel={(1-iv/ic)*100:.1f}%')
print('\n-> conclusao ViT<CNN se mantem em ambas as variantes? '
      f"{'SIM' if inconsistent_tws(y_vit).mean() < inconsistent_tws(y_cnn).mean() else 'NAO'}")


print()
print('=' * 70)
print('B3. CALIBRACAO (ECE + reliability)')
print('=' * 70)

def ece(y_t, proba, n_bins=15):
    """Expected Calibration Error multi-rotulo: trata cada (amostra,label) como
    uma predicao binaria; confianca = max(p,1-p), acerto = (pred==alvo)."""
    yt = y_t.ravel()
    p = proba.ravel()
    pred = (p > 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == yt).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e, total = 0.0, len(p)
    curve = []
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        acc_b = correct[m].mean()
        conf_b = conf[m].mean()
        e += (m.sum() / total) * abs(acc_b - conf_b)
        curve.append((conf_b, acc_b, m.sum()))
    return e, curve

ece_cnn, curve_cnn = ece(y_true, proba_cnn)
ece_vit, curve_vit = ece(y_true, proba_vit)
print(f'\nECE CNN = {ece_cnn*100:.2f}%')
print(f'ECE ViT = {ece_vit*100:.2f}%')
print(f'-> {"ViT" if ece_vit < ece_cnn else "CNN"} melhor calibrado '
      f'(menor ECE), reducao relativa {abs(1-ece_vit/ece_cnn)*100:.1f}%')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    for curve, name, mk in [(curve_cnn, f'MobileNetV2 (ECE={ece_cnn*100:.2f}%)', 'o-'),
                            (curve_vit, f'ViT-B/16 (ECE={ece_vit*100:.2f}%)', 's-')]:
        cx = [c[0] for c in curve]
        cy = [c[1] for c in curve]
        ax.plot(cx, cy, mk, ms=4, label=name)
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram'); ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'reliability_diagram.png', dpi=150)
    print(f'  reliability_diagram.png salvo em {OUT_DIR}')
except Exception as ex:
    print(f'  (figura nao gerada: {ex})')


print()
print('=' * 70)
print('B4. LATENCIA: ESTATISTICA ROBUSTA (n=50)')
print('=' * 70)

def boot_ci(x, stat=np.median, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])

for name, x in [('CNN', lat_cnn), ('ViT', lat_vit)]:
    lo, hi = boot_ci(x)
    print(f'\n{name}: mean={x.mean():.2f}  median={np.median(x):.2f}  '
          f'IQR=[{np.percentile(x,25):.2f},{np.percentile(x,75):.2f}]')
    print(f'  p95={np.percentile(x,95):.2f}  p99={np.percentile(x,99):.2f}  '
          f'IC95%(mediana)=[{lo:.2f},{hi:.2f}]')

# diferenca de medianas com IC bootstrap
rng = np.random.default_rng(42)
diffs = [np.median(rng.choice(lat_vit, len(lat_vit), True)) -
         np.median(rng.choice(lat_cnn, len(lat_cnn), True)) for _ in range(10000)]
dlo, dhi = np.percentile(diffs, [2.5, 97.5])
print(f'\nDiferenca de medianas (ViT-CNN): {np.median(lat_vit)-np.median(lat_cnn):.2f} ms'
      f'  IC95%=[{dlo:.2f},{dhi:.2f}] -> {"exclui 0 (robusto)" if dlo>0 else "inclui 0"}')
