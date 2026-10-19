"""Final paper figures from round-1 and round-2 outputs (all numbers read from files).

Usage: python make_figures_round2.py
Outputs: out/paper_figs/*.png, out/paper_figs/gradcam_hybrid.json
"""
import glob, json, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FULL = os.path.join(HERE, 'out'), os.path.join(HERE, 'out', 'full')
FIG = os.path.join(OUT, 'paper_figs')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9, 'savefig.dpi': 300})
BLUE, GREY, ORANGE, GREEN, LIGHT = '#2f5f8f', '#8c8c8c', '#c8702a', '#4f8a4f', '#9fb7cf'


def seed_avg(pattern):
    runs = [pd.read_csv(f) for f in sorted(glob.glob(pattern))]
    d = runs[0][['id', 'label']].copy()
    d['prob'] = np.mean([r.prob.values for r in runs], 0)
    return d, [roc_auc_score(r.label, r.prob) for r in runs]


def main():
    hyb, _ = seed_avg(os.path.join(FULL, 'preds_full_2dD_phylop_s*_test.csv'))
    vep = pd.read_csv(os.path.join(OUT, 'vep_scores.csv'))

    # ---- ROC on the subset scored by all three tools ----
    v = hyb.merge(vep, on='id')
    c = v[v.sift.notna() & v.polyphen.notna() & v.cadd_phred.notna()]
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    for name, s, col in [('CADD', c.cadd_phred, ORANGE), ('Hybrid CNN', c.prob, BLUE),
                         ('PolyPhen-2', c.polyphen, GREEN), ('SIFT', 1 - c.sift, GREY)]:
        f, t, _ = roc_curve(c.label, s)
        ax.plot(f, t, color=col, lw=1.4, label=f'{name} (AUC {roc_auc_score(c.label, s):.3f})')
    ax.plot([0, 1], [0, 1], ls=':', color='black', lw=0.8)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.legend(frameon=False, fontsize=7.5, loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_roc_baselines.png')); plt.close(fig)

    # ---- model progression / ablation bars (test ROC-AUC, mean +/- SD over 3 seeds) ----
    def aucs(pattern):
        return [roc_auc_score(r.label, r.prob) for r in (pd.read_csv(f) for f in sorted(glob.glob(pattern)))]
    bars = [('2D, 56k\nrandom split', aucs(os.path.join(OUT, 'preds_random_2dD_s*.csv')), LIGHT),
            ('1D, 56k', aucs(os.path.join(OUT, 'preds_tuned_1D-C_s*.csv')), GREY),
            ('2D, 56k', aucs(os.path.join(OUT, 'preds_gene_2dD_s*.csv')), GREY),
            ('2D, 432k', aucs(os.path.join(FULL, 'preds_full_2dD_s*_test.csv')), ORANGE),
            ('2D + phyloP\n432k', aucs(os.path.join(FULL, 'preds_full_2dD_phylop_s*_test.csv')), BLUE)]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, (lab, a, col) in enumerate(bars):
        m, sd = np.mean(a), np.std(a, ddof=1)
        ax.bar(i, m, color=col, edgecolor='black', lw=0.7, width=0.62)
        ax.errorbar(i, m, yerr=sd, color='black', capsize=2.5, lw=0.9)
        ax.text(i, m + sd + 0.006, f'{m:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=7)
    ax.set_ylim(0.8, 1.01); ax.set_ylabel('Test ROC-AUC (mean ± SD)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_model_progression.png')); plt.close(fig)

    # ---- confusion matrix (seed-averaged hybrid) ----
    cm = confusion_matrix(hyb.label, (hyb.prob >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    ax.imshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val:,}', ha='center', va='center', color='white' if val > cm.max() / 2 else 'black', fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Benign', 'Pathogenic'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Benign', 'Pathogenic'])
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_confusion.png')); plt.close(fig)

    # ---- loss curves (hybrid seed 0) ----
    h = json.load(open(os.path.join(FULL, 'history_full_2dD_phylop_s0.json')))
    e = np.arange(1, len(h['loss']) + 1)
    best = int(np.argmin(h['val_loss'])) + 1
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(e, h['loss'], color=BLUE, label='Training loss')
    ax.plot(e, h['val_loss'], color=ORANGE, label='Validation loss (held-out genes)')
    ax.axvline(best, ls=':', color='black', lw=0.8)
    ax.text(best - 0.3, min(h['loss']) * 1.05, f'best epoch {best}', fontsize=7, ha='right')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Binary cross-entropy')
    ax.legend(frameon=False, fontsize=7.5, loc='upper center'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_loss.png')); plt.close(fig)

    # ---- Grad-CAM on the sequence trunk of hybrid seed 0 (last Conv2D layer) ----
    meta = pd.read_csv(os.path.join(FULL, 'variants_full.csv'), low_memory=False)
    d = np.load(os.path.join(FULL, 'dataset_full.npz'))
    P = np.load(os.path.join(FULL, 'phylop_full.npy'))
    idx = np.where(meta.split.values == 'test')[0]
    X = d['X'][idx].astype(np.float32)
    r, a = X[:, :4].transpose(0, 2, 1), X[:, 4:].transpose(0, 2, 1)
    seq = np.stack([r, a, a - r], 1)
    ph = P[idx][..., None]
    y = d['y'][idx]
    model = keras.models.load_model(os.path.join(FULL, 'model_full_2dD_phylop_s0.keras'))
    conv = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)][-1]
    gm = keras.Model(model.inputs, [conv.output, model.output])
    profs = []
    for i in range(0, len(idx), 512):
        inp = {'seq': tf.convert_to_tensor(seq[i:i + 512]), 'phylop': tf.convert_to_tensor(ph[i:i + 512])}
        with tf.GradientTape() as tape:
            fmap, pred = gm(inp, training=False)
            score = pred[:, 0]
        g = tape.gradient(score, fmap)
        w = tf.reduce_mean(g, axis=(1, 2))
        cam = tf.nn.relu(tf.einsum('bhwc,bc->bhw', fmap, w))
        cam = tf.reduce_mean(cam, axis=1).numpy()
        cam = np.stack([np.interp(np.linspace(0, cam.shape[1] - 1, 101), np.arange(cam.shape[1]), c_) for c_ in cam])
        profs.append(cam / (cam.max(1, keepdims=True) + 1e-8))
    prof = np.concatenate(profs)
    pos = np.arange(-50, 51)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for cls, col, name in [(1, BLUE, 'Pathogenic'), (0, GREY, 'Benign')]:
        ax.plot(pos, prof[y == cls].mean(0), color=col, label=name)
    ax.axvline(0, ls=':', color='black', lw=0.8)
    ax.set_xlabel('Position relative to variant (bp)'); ax.set_ylabel('Mean Grad-CAM (normalized)')
    ax.legend(frameon=False, fontsize=7.5); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_gradcam_profile.png')); plt.close(fig)
    central = prof[:, 40:61].mean(1) / (prof.mean(1) + 1e-8)
    json.dump({'layer': conv.name, 'n_test': int(len(idx)),
               'central_21bp_over_window_mean': round(float(central.mean()), 3),
               'peak_offset_pathogenic': int(pos[prof[y == 1].mean(0).argmax()]),
               'peak_offset_benign': int(pos[prof[y == 0].mean(0).argmax()]),
               'bar_values': {b[0].replace('\n', ' '): [round(float(np.mean(b[1])), 4), round(float(np.std(b[1], ddof=1)), 4)]
                              for b in bars},
               'confusion_seed_avg': cm.tolist(), 'hybrid_loss_best_epoch': best, 'hybrid_loss_epochs': len(e)},
              open(os.path.join(FIG, 'figure_values.json'), 'w'), indent=2)
    print(open(os.path.join(FIG, 'figure_values.json')).read())


if __name__ == '__main__':
    main()
