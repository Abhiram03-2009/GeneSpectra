"""Round 2 evaluation (TUNING_PROTOCOL.md, round 2).

1. Validation/test predictions for the three candidate primary models:
   r1_2dD (round-1 training set), full_2dD (R2-1), full_2dD_phylop (R2-3)
2. Primary model = highest mean validation AUC over seeds 0-2.
3. Test metrics for every candidate; per-consequence and missense results; baseline comparison.
4. R2-2: logistic regression on the validation set, CADD alone vs CADD + CNN logit, evaluated on the test set.

Usage: python evaluate_round2.py
Outputs: out/full/results_round2.json, out/full/fig_*.png
"""
import glob, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FULL = os.path.join(HERE, 'out'), os.path.join(HERE, 'out', 'full')


def r1_val_preds():
    """Round-1 2D-D models were saved; predict the validation set with them (test preds already exist)."""
    files = sorted(glob.glob(os.path.join(FULL, 'preds_r1_2dD_s*_val.csv')))
    if len(files) == 3:
        return
    from tensorflow import keras
    from tune_encodings import in_3x101x4
    d = np.load(os.path.join(OUT, 'dataset.npz'))
    meta = pd.read_csv(os.path.join(OUT, 'variants.csv'))
    for split in ('val', 'test'):
        idx = np.where(meta.split_gene.values == split)[0]
        Xi = in_3x101x4(d['X'][idx])
        for s in range(3):
            m = keras.models.load_model(os.path.join(OUT, f'model_gene_2dD_s{s}.keras'))
            p = m.predict(Xi, batch_size=1024, verbose=0)[:, 0]
            pd.DataFrame({'id': meta.id.values[idx], 'label': d['y'][idx].astype(int), 'prob': p}) \
                .to_csv(os.path.join(FULL, f'preds_r1_2dD_s{s}_{split}.csv'), index=False)


def load(exp, split):
    return [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(FULL, f'preds_{exp}_s*_{split}.csv')))]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    r1_val_preds()
    meta = pd.read_csv(os.path.join(OUT, 'variants.csv'), dtype={'chrom': str})
    res = {'candidates': {}}
    for exp in ('r1_2dD', 'full_2dD', 'full_2dD_phylop'):
        val, test = load(exp, 'val'), load(exp, 'test')
        if len(val) < 3 or len(test) < 3:
            continue
        va = [roc_auc_score(v.label, v.prob) for v in val]
        ta = [roc_auc_score(t.label, t.prob) for t in test]
        avg = test[0][['id', 'label']].copy()
        avg['prob'] = np.mean([t.prob.values for t in test], 0)
        res['candidates'][exp] = dict(
            val_auc_mean=round(float(np.mean(va)), 4), val_auc_per_seed=[round(float(a), 4) for a in va],
            test_auc_mean_sd=[round(float(np.mean(ta)), 4), round(float(np.std(ta, ddof=1)), 4)],
            test_seed0=E.with_ci(test[0].label.values, test[0].prob.values, (test[0].prob.values >= 0.5).astype(int)),
            test_seed_avg=E.with_ci(avg.label.values, avg.prob.values, (avg.prob.values >= 0.5).astype(int)))
    primary = max(res['candidates'], key=lambda k: res['candidates'][k]['val_auc_mean'])
    res['primary'] = primary
    print('primary by validation AUC:', primary, {k: v['val_auc_mean'] for k, v in res['candidates'].items()})

    # pairwise paired comparisons on the test set (seed-averaged probabilities)
    def avgp(exp, split):
        runs = load(exp, split)
        df = runs[0][['id', 'label']].copy(); df['prob'] = np.mean([r.prob.values for r in runs], 0)
        return df
    T = {k: avgp(k, 'test') for k in res['candidates']}
    V = {k: avgp(k, 'val') for k in res['candidates']}
    y = T[primary].label.values
    res['paired_test_auc'] = {f'{a}_minus_{b}': E.paired_auc_diff(y, T[a].prob.values, T[b].prob.values)
                              for a in T for b in T if a < b}

    # per-consequence / missense for primary (seed-averaged)
    g = T[primary].merge(meta[['id', 'consequence', 'vtype', 'new_since_2023']], on='id')
    pc = []
    for c, sub in g.groupby('consequence'):
        if len(sub) >= 40 and sub.label.nunique() == 2:
            pc.append(dict(consequence=c, n=len(sub), pathogenic=int(sub.label.sum()),
                           roc_auc=round(roc_auc_score(sub.label, sub.prob), 4)))
    res['primary_per_consequence'] = sorted(pc, key=lambda r: -r['n'])
    t = g[g.new_since_2023]
    res['primary_temporal'] = E.with_ci(t.label.values, t.prob.values, (t.prob.values >= 0.5).astype(int))
    for exp in T:   # missense AUC for every candidate
        gm = T[exp].merge(meta[['id', 'consequence']], on='id')
        gm = gm[gm.consequence == 'missense_variant']
        res['candidates'][exp]['missense_test_auc'] = round(float(roc_auc_score(gm.label, gm.prob)), 4)

    # baselines vs primary on identical subsets
    vt = pd.read_csv(os.path.join(OUT, 'vep_scores.csv'))
    gb = g.merge(vt, on='id', how='left')
    allc = gb[gb.sift.notna() & gb.polyphen.notna() & gb.cadd_phred.notna()]
    ya = allc.label.values
    common = {'n': len(allc), 'CNN': E.with_ci(ya, allc.prob.values, (allc.prob.values >= 0.5).astype(int))}
    for name, (col, rule, _) in E.THRESH.items():
        sc = allc[col].values.astype(float)
        common[name] = E.with_ci(ya, E.CONT[name](sc), rule(sc).astype(int))
        common['cnn_minus_' + name] = E.paired_auc_diff(ya, allc.prob.values, E.CONT[name](sc))
    res['baselines_common_subset_primary'] = common

    # ---------- R2-2: CADD alone vs CADD + CNN (fit on validation, evaluate on test) ----------
    vv_path = os.path.join(OUT, 'vep_scores_val.csv')
    if os.path.exists(vv_path):
        def feats(df, vep, with_cnn):
            m = df.merge(vep[['id', 'cadd_phred']], on='id', how='left')
            miss = m.cadd_phred.isna().values.astype(float)
            cadd = m.cadd_phred.fillna(0).values
            cols = [cadd, miss] + ([logit(m.prob.values)] if with_cnn else [])
            return np.column_stack(cols), m.label.values, ~m.cadd_phred.isna().values
        vv = pd.read_csv(vv_path)
        stack = {}
        preds = {}
        for name, wc in (('CADD_only', False), ('CADD_plus_CNN', True)):
            Xv, yv, _ = feats(V[primary], vv, wc)
            Xt, yt, has = feats(T[primary], vt, wc)
            lr = LogisticRegression(max_iter=1000).fit(Xv, yv)
            pt = lr.predict_proba(Xt)[:, 1]
            preds[name] = (pt, yt, has)
            stack[name] = {'all_test': E.with_ci(yt, pt, (pt >= 0.5).astype(int)),
                           'cadd_scored_subset': E.with_ci(yt[has], pt[has], (pt[has] >= 0.5).astype(int)),
                           'coef': [round(float(c), 4) for c in lr.coef_[0]]}
        (pa, yt, has), (pb, _, _) = preds['CADD_only'], preds['CADD_plus_CNN']
        stack['plus_minus_only_all_test'] = E.paired_auc_diff(yt, pb, pa)
        stack['plus_minus_only_cadd_subset'] = E.paired_auc_diff(yt[has], pb[has], pa[has])
        gm = T[primary].merge(meta[['id', 'consequence']], on='id')
        mis = (gm.consequence == 'missense_variant').values & has
        stack['missense_cadd_subset'] = {'n': int(mis.sum()),
                                         'CADD_only_auc': round(float(roc_auc_score(yt[mis], pa[mis])), 4),
                                         'CADD_plus_CNN_auc': round(float(roc_auc_score(yt[mis], pb[mis])), 4),
                                         'diff': E.paired_auc_diff(yt[mis], pb[mis], pa[mis])}
        nocadd = ~has
        if nocadd.sum() > 50 and len(set(yt[nocadd])) == 2:
            stack['no_cadd_score_subset'] = {'n': int(nocadd.sum()),
                                             'CNN_auc': round(float(roc_auc_score(yt[nocadd], T[primary].prob.values[nocadd])), 4)}
        res['cadd_stacking'] = stack

        fig, ax = plt.subplots(figsize=(3.3, 3.0))
        for name, s, c in [('CADD + CNN', pb, E.BLUE), ('CADD alone', pa, E.ORANGE)]:
            f, tp_, _ = roc_curve(yt[has], s[has])
            ax.plot(f, tp_, color=c, lw=1.4, label=f'{name} (AUC {roc_auc_score(yt[has], s[has]):.3f})')
        f, tp_, _ = roc_curve(yt[has], T[primary].prob.values[has])
        ax.plot(f, tp_, color=E.GREY, lw=1.2, label=f'CNN alone (AUC {roc_auc_score(yt[has], T[primary].prob.values[has]):.3f})')
        ax.plot([0, 1], [0, 1], ls=':', color='black', lw=0.8)
        ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
        ax.legend(frameon=False, fontsize=7.5, loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(FULL, 'fig_roc_cadd_stacking.png')); plt.close(fig)

    res['prep_full'] = json.load(open(os.path.join(FULL, 'prep_full_summary.json')))
    ps = os.path.join(FULL, 'phylop_summary.json')
    if os.path.exists(ps):
        res['phylop'] = json.load(open(ps))
    res['train_full'] = json.load(open(os.path.join(FULL, 'train_full_summary.json')))

    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list): return [clean(v) for v in o]
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, (np.floating, float)): return None if np.isnan(o) else round(float(o), 4)
        return o
    json.dump(clean(res), open(os.path.join(FULL, 'results_round2.json'), 'w'), indent=2)
    print('saved results_round2.json')


if __name__ == '__main__':
    main()
