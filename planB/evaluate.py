"""Plan B, step 4: metrics with bootstrap CIs, baseline comparison on identical variants, figures.

Usage: python evaluate.py
Outputs: out/results.json, out/fig_*.png, out/per_consequence.csv
"""
import glob, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
                             accuracy_score, matthews_corrcoef, confusion_matrix, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
B = 1000
PRIMARY = 'gene_2dD'          # tuned 2D design selected on validation AUC (TUNING_PROTOCOL.md)
rng = np.random.default_rng(42)
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9, 'savefig.dpi': 300})
BLUE, GREY, ORANGE, GREEN = '#2f5f8f', '#8c8c8c', '#c8702a', '#4f8a4f'

# thresholds for binarizing baseline scores (standard published cut-offs)
THRESH = {'SIFT': ('sift', lambda s: s < 0.05, 'score < 0.05 (deleterious)'),
          'PolyPhen-2': ('polyphen', lambda s: s > 0.446, 'score > 0.446 (possibly/probably damaging, HumVar)'),
          'CADD': ('cadd_phred', lambda s: s >= 20, 'PHRED >= 20')}
CONT = {'SIFT': lambda s: 1 - s, 'PolyPhen-2': lambda s: s, 'CADD': lambda s: s}


def point(y, prob, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), n_pathogenic=int(y.sum()), n_benign=int((1 - y).sum()),
                accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), specificity=tn / (tn + fp) if tn + fp else float('nan'),
                f1=f1_score(y, pred), mcc=matthews_corrcoef(y, pred),
                roc_auc=roc_auc_score(y, prob), pr_auc=average_precision_score(y, prob),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def with_ci(y, prob, pred):
    est = point(y, prob, pred)
    keys = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'mcc', 'roc_auc', 'pr_auc']
    boots = {k: [] for k in keys}
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        p = point(y[i], prob[i], pred[i])
        for k in keys:
            boots[k].append(p[k])
    for k in keys:
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        est[k] = [round(float(est[k]), 4), round(float(lo), 4), round(float(hi), 4)]
    return est


def paired_auc_diff(y, a, b):
    """AUC(a) - AUC(b) on identical variants, with paired bootstrap 95% CI and two-sided p-value."""
    d0 = roc_auc_score(y, a) - roc_auc_score(y, b)
    ds = []
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        ds.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    ds = np.array(ds)
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return dict(diff=round(float(d0), 4), ci=[round(float(x), 4) for x in np.percentile(ds, [2.5, 97.5])],
                p_bootstrap=float(max(p, 1 / len(ds))))


def load_preds(exp):
    files = sorted(glob.glob(os.path.join(OUT, f'preds_{exp}_s*.csv')))
    return [pd.read_csv(f) for f in files]


def main():
    meta = pd.read_csv(os.path.join(OUT, 'variants.csv'), dtype={'chrom': str})
    res = {'prep': json.load(open(os.path.join(OUT, 'prep_summary.json'))),
           'train': json.load(open(os.path.join(OUT, 'train_summary.json')))}

    # ---------- CNN experiments ----------
    res['experiments'] = {}
    for exp in ('gene_2dD', 'random_2dD', 'tuned_1D-C', 'gene_2d', 'gene_1d', 'random_2d'):
        runs = load_preds(exp)
        if not runs:
            continue
        r0 = runs[0]
        y, p = r0.label.values, r0.prob.values
        entry = {'seed0': with_ci(y, p, (p >= 0.5).astype(int))}
        per_seed = [point(r.label.values, r.prob.values, (r.prob.values >= 0.5).astype(int)) for r in runs]
        entry['across_seeds'] = {k: [round(float(np.mean([s[k] for s in per_seed])), 4),
                                     round(float(np.std([s[k] for s in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0, 4)]
                                 for k in ('accuracy', 'f1', 'mcc', 'roc_auc', 'pr_auc')}
        entry['n_seeds'] = len(runs)
        res['experiments'][exp] = entry

    g = load_preds(PRIMARY)[0].merge(meta[['id', 'consequence', 'vtype', 'new_since_2023', 'gene']], on='id')
    y, p = g.label.values, g.prob.values

    # ---------- ablation and leakage, paired where possible ----------
    for other, key in (('tuned_1D-C', 'ablation_2dD_minus_1dC_auc'), ('gene_2d', 'ablation_2dD_minus_original2d_auc')):
        o = load_preds(other)
        if o:
            assert (o[0].id.values == g.id.values).all()
            res[key] = {'seed0': paired_auc_diff(y, p, o[0].prob.values)}
            a_runs = load_preds(PRIMARY)
            if len(a_runs) == len(o):
                res[key]['seed_avg_prob'] = paired_auc_diff(y, np.mean([r.prob for r in a_runs], 0),
                                                            np.mean([r.prob for r in o], 0))
    r_runs = load_preds('random_2dD')
    if r_runs:
        res['leakage_random_minus_gene_auc_mean'] = round(
            float(np.mean([roc_auc_score(r.label, r.prob) for r in r_runs]) -
                  np.mean([roc_auc_score(r.label, r.prob) for r in load_preds(PRIMARY)])), 4)

    # ---------- temporal subset ----------
    t = g[g.new_since_2023]
    if t.label.nunique() == 2:
        res['temporal_primary'] = with_ci(t.label.values, t.prob.values, (t.prob.values >= 0.5).astype(int))

    # ---------- per-consequence error analysis ----------
    rows = []
    for c, sub in g.groupby('consequence'):
        pred = (sub.prob >= 0.5).astype(int)
        rows.append(dict(consequence=c, n=len(sub), pathogenic=int(sub.label.sum()), benign=int((1 - sub.label).sum()),
                         accuracy=round(accuracy_score(sub.label, pred), 4),
                         fn=int(((pred == 0) & (sub.label == 1)).sum()), fp=int(((pred == 1) & (sub.label == 0)).sum()),
                         roc_auc=round(roc_auc_score(sub.label, sub.prob), 4) if sub.label.nunique() == 2 else None))
    pc = pd.DataFrame(rows).sort_values('n', ascending=False)
    pc.to_csv(os.path.join(OUT, 'per_consequence.csv'), index=False)
    res['per_consequence'] = pc.to_dict('records')
    vt = []
    for c, sub in g.groupby('vtype'):
        pred = (sub.prob >= 0.5).astype(int)
        vt.append(dict(vtype=c, n=len(sub), accuracy=round(accuracy_score(sub.label, pred), 4),
                       roc_auc=round(roc_auc_score(sub.label, sub.prob), 4) if sub.label.nunique() == 2 else None))
    res['per_variant_type'] = vt

    # ---------- baselines ----------
    vep_path = os.path.join(OUT, 'vep_scores.csv')
    if os.path.exists(vep_path):
        v = g.merge(pd.read_csv(vep_path), on='id', how='left')
        res['vep_info'] = json.load(open(os.path.join(OUT, 'vep_info.json')))
        res['baselines'] = {'thresholds': {k: t_[2] for k, t_ in THRESH.items()}, 'per_tool': {}}
        for name, (col, rule, _) in THRESH.items():
            cov = v[v[col].notna()]
            yy = cov.label.values
            sc = cov[col].values.astype(float)
            ent = dict(coverage=dict(n=len(cov), frac=round(len(cov) / len(v), 4),
                                     pathogenic=int(yy.sum()), benign=int((1 - yy).sum()),
                                     snv=int((cov.vtype == 'single_nucleotide_variant').sum()),
                                     indel=int((cov.vtype != 'single_nucleotide_variant').sum())))
            if len(set(yy)) == 2:
                ent['tool'] = with_ci(yy, CONT[name](sc), rule(sc).astype(int))
                ent['cnn_same_variants'] = with_ci(yy, cov.prob.values, (cov.prob.values >= 0.5).astype(int))
                ent['cnn_minus_tool_auc'] = paired_auc_diff(yy, cov.prob.values, CONT[name](sc))
            res['baselines']['per_tool'][name] = ent
        # identical subset scored by all three tools
        allc = v[v.sift.notna() & v.polyphen.notna() & v.cadd_phred.notna()]
        ya = allc.label.values
        common = dict(n=len(allc), pathogenic=int(ya.sum()), benign=int((1 - ya).sum()), methods={})
        if len(set(ya)) == 2:
            common['methods']['CNN'] = with_ci(ya, allc.prob.values, (allc.prob.values >= 0.5).astype(int))
            for name, (col, rule, _) in THRESH.items():
                sc = allc[col].values.astype(float)
                common['methods'][name] = with_ci(ya, CONT[name](sc), rule(sc).astype(int))
                common['cnn_minus_' + name] = paired_auc_diff(ya, allc.prob.values, CONT[name](sc))
        res['baselines']['common_subset'] = common

        # figure: ROC on the common subset
        if len(set(ya)) == 2:
            fig, ax = plt.subplots(figsize=(3.3, 3.0))
            for name, col, s, c in [('2D CNN', 'prob', allc.prob.values, BLUE),
                                    ('CADD', 'cadd_phred', allc.cadd_phred.values, ORANGE),
                                    ('PolyPhen-2', 'polyphen', allc.polyphen.values, GREEN),
                                    ('SIFT', 'sift', 1 - allc.sift.values, GREY)]:
                fpr, tpr, _ = roc_curve(ya, s)
                ax.plot(fpr, tpr, color=c, lw=1.4, label=f'{name} (AUC {roc_auc_score(ya, s):.3f})')
            ax.plot([0, 1], [0, 1], ls=':', color='black', lw=0.8)
            ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
            ax.legend(frameon=False, fontsize=7.5, loc='lower right')
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_roc_baselines.png')); plt.close(fig)

    # ---------- figure: experiment comparison (ROC-AUC with CI) ----------
    ex = res['experiments']
    labels = [('random_2dD', '2D CNN\nrandom split'), ('gene_2dD', '2D CNN\ngene-disjoint'),
              ('tuned_1D-C', '1D CNN\ngene-disjoint')]
    labels = [(k, l) for k, l in labels if k in ex]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for i, (k, l) in enumerate(labels):
        est, sd = ex[k]['across_seeds']['roc_auc']; lo, hi = est - sd, est + sd
        ax.bar(i, est, color=[GREY, BLUE, ORANGE][['random_2dD', 'gene_2dD', 'tuned_1D-C'].index(k)],
               edgecolor='black', lw=0.8, width=0.6)
        ax.errorbar(i, est, yerr=[[est - lo], [hi - est]], color='black', capsize=3, lw=1)
        ax.text(i, hi + 0.01, f'{est:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([l for _, l in labels], fontsize=8)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel('Test ROC-AUC (mean ± SD, 3 seeds)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_split_ablation.png')); plt.close(fig)

    # ---------- figure: confusion matrix (gene_2d seed 0) ----------
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    ax.imshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val:,}', ha='center', va='center', color='white' if val > cm.max() / 2 else 'black', fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Benign', 'Pathogenic'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Benign', 'Pathogenic'])
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_confusion_primary.png')); plt.close(fig)

    # ---------- figure: loss curves ----------
    h = json.load(open(os.path.join(OUT, f'history_{PRIMARY}_s0.json')))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    e = np.arange(1, len(h['loss']) + 1)
    ax.plot(e, h['loss'], color=BLUE, label='Training loss')
    ax.plot(e, h['val_loss'], color=ORANGE, label='Validation loss (held-out genes)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Binary cross-entropy')
    ax.legend(frameon=False, fontsize=7.5); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_loss_primary.png')); plt.close(fig)

    # ---------- figure: Grad-CAM positional profile ----------
    gc_path = os.path.join(OUT, f'gradcam_{PRIMARY}_s0.npz')
    if os.path.exists(gc_path):
        gc = np.load(gc_path)
        prof, lab = gc['profile'], gc['label']
        pos = np.arange(-50, 51)
        fig, ax = plt.subplots(figsize=(3.3, 2.2))
        for cls, c, name in [(1, BLUE, 'Pathogenic'), (0, GREY, 'Benign')]:
            m = prof[lab == cls].mean(0)
            ax.plot(pos, m, color=c, label=name)
        ax.axvline(0, ls=':', color='black', lw=0.8)
        ax.set_xlabel('Position relative to variant (bp)'); ax.set_ylabel('Mean Grad-CAM (norm.)')
        ax.legend(frameon=False, fontsize=7.5); ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_gradcam_profile.png')); plt.close(fig)
        central = prof[:, 40:61].mean(1) / (prof.mean(1) + 1e-8)
        res['gradcam'] = dict(layer=str(gc['layer']),
                              central_21bp_to_mean_ratio=round(float(central.mean()), 3),
                              peak_offset_pathogenic=int(pos[prof[lab == 1].mean(0).argmax()]),
                              peak_offset_benign=int(pos[prof[lab == 0].mean(0).argmax()]))

    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list): return [clean(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, float)): return None if np.isnan(o) else round(float(o), 4)
        return o
    json.dump(clean(res), open(os.path.join(OUT, 'results.json'), 'w'), indent=2)
    print(json.dumps(clean({k: res[k] for k in res if k not in ('prep', 'train', 'vep_info', 'per_consequence')}), indent=1)[:6000])


if __name__ == '__main__':
    main()
