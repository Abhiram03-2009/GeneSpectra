"""Round 2 (R2-1): enlarge the training set; keep round-1 validation and test sets exactly.

Usage: python prep_full.py
Outputs: out/full/dataset_full.npz, out/full/variants_full.csv, out/full/prep_full_summary.json
"""
import json, os
import numpy as np
import pandas as pd
import twobitreader
import prep_data as P

OUT_FULL = os.path.join(P.OUT, 'full')
os.makedirs(OUT_FULL, exist_ok=True)


def windows(df, tb):
    idx = {b: i for i, b in enumerate('ACGT')}
    X = np.zeros((len(df), 8, P.W), dtype=np.uint8)
    keep = np.ones(len(df), dtype=bool)
    for i, r in enumerate(df.itertuples()):
        seq = tb['chr' + r.chrom]
        p0 = r.pos - 1
        left = seq[p0 - P.FLANK:p0].upper()
        if seq[p0:p0 + len(r.ref)].upper() != r.ref or len(left) != P.FLANK:
            keep[i] = False; continue
        ref_win = (left + seq[p0:p0 + P.W].upper())[:P.W]
        alt_win = (left + r.alt + seq[p0 + len(r.ref):p0 + len(r.ref) + P.W].upper())[:P.W]
        for j, b in enumerate(ref_win):
            if b in idx: X[i, idx[b], j] = 1
        for j, b in enumerate(alt_win):
            if b in idx: X[i, 4 + idx[b], j] = 1
    return X, keep


def main():
    r1 = pd.read_csv(os.path.join(P.OUT, 'variants.csv'), dtype={'chrom': str})
    d1 = np.load(os.path.join(P.OUT, 'dataset.npz'))
    hold = r1[r1.split_gene.isin(['val', 'test'])]
    hold_genes = set(hold.gene)

    rows = []
    for chrom, pos, vid, ref, alt, info in P.read_vcf(os.path.join(P.DATA, 'clinvar.vcf.gz')):
        y = P.label_of(info)
        if y is None or info.get('CLNREVSTAT', '') not in P.STAR1 or chrom not in P.CHROMS \
                or info.get('CLNVC', '') not in P.TYPES or not (P.ACGT.match(ref) and P.ACGT.match(alt)) \
                or abs(len(ref) - len(alt)) > P.MAX_INDEL or max(len(ref), len(alt)) > P.MAX_INDEL + 1:
            continue
        gi = info.get('GENEINFO')
        if not gi:
            continue
        gene = gi.split('|')[0].split(':')[0]
        if gene in hold_genes:
            continue
        mc = info.get('MC', '')
        mc = mc.split(',')[0].split('|')[1] if '|' in mc else 'unknown'
        rows.append((vid, chrom, pos, ref, alt, y, gene, info['CLNVC'], mc))
    tr = pd.DataFrame(rows, columns=['id', 'chrom', 'pos', 'ref', 'alt', 'label', 'gene', 'vtype', 'consequence'])
    tr = tr.drop_duplicates(['chrom', 'pos', 'ref', 'alt'])
    rng = np.random.default_rng(P.SEED)
    path, ben = tr[tr.label == 1], tr[tr.label == 0]
    ben = ben.iloc[rng.choice(len(ben), min(len(path), len(ben)), replace=False)]
    tr = pd.concat([path, ben]).sample(frac=1, random_state=P.SEED).reset_index(drop=True)

    tb = twobitreader.TwoBitFile(os.path.join(P.DATA, 'hg38.2bit'))
    Xtr, keep = windows(tr, tb)
    tr, Xtr = tr[keep].reset_index(drop=True), Xtr[keep]
    tr['split'] = 'train'

    hold_idx = np.where(r1.split_gene.isin(['val', 'test']).values)[0]
    h = r1.iloc[hold_idx][['id', 'chrom', 'pos', 'ref', 'alt', 'label', 'gene', 'vtype', 'consequence',
                           'split_gene', 'new_since_2023']].rename(columns={'split_gene': 'split'})
    df = pd.concat([tr, h], ignore_index=True)
    X = np.concatenate([Xtr, d1['X'][hold_idx]])
    summ = {s: {'n': int((df.split == s).sum()), 'pathogenic': int(df[df.split == s].label.sum()),
                'benign': int((df[df.split == s].label == 0).sum()), 'genes': int(df[df.split == s].gene.nunique())}
            for s in ('train', 'val', 'test')}
    summ['ref_mismatch_dropped'] = int((~keep).sum())
    summ['gene_overlap_train_vs_heldout'] = len(set(tr.gene) & hold_genes)
    np.savez(os.path.join(OUT_FULL, 'dataset_full.npz'), X=X, y=df.label.values.astype(np.uint8))
    df.to_csv(os.path.join(OUT_FULL, 'variants_full.csv'), index=False)
    json.dump(summ, open(os.path.join(OUT_FULL, 'prep_full_summary.json'), 'w'), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == '__main__':
    main()
