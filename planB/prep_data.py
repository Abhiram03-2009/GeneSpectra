"""Plan B, step 1: build the ClinVar variant dataset and the train/val/test splits.

Inputs (in data/): clinvar.vcf.gz (current GRCh38 release), clinvar_20230107.vcf.gz, hg38.2bit
Outputs (in out/): dataset.npz (one-hot windows + labels), variants.csv (metadata + split assignments),
                   prep_summary.json (all counts, for the paper)

Usage: python prep_data.py
"""
import gzip, json, os, re
import numpy as np
import pandas as pd
import twobitreader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

SEED = 42
FLANK = 50                 # +/-50 bp -> 101 bp window
W = 2 * FLANK + 1
MAX_INDEL = 10             # small indels only: |len(REF) - len(ALT)| <= 10 and both alleles <= 11 bp
N_PER_CLASS = 40_000       # balanced sample size per class (keeps CPU training time practical)

PATH_LABELS = {'Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic'}
BEN_LABELS = {'Benign', 'Likely_benign', 'Benign/Likely_benign'}
# review status >= 1 star (ClinVar star mapping)
STAR1 = {'criteria_provided,_single_submitter', 'criteria_provided,_multiple_submitters,_no_conflicts',
         'reviewed_by_expert_panel', 'practice_guideline'}
TYPES = {'single_nucleotide_variant', 'Deletion', 'Insertion', 'Indel', 'Duplication'}
CHROMS = {str(i) for i in range(1, 23)} | {'X', 'Y'}
ACGT = re.compile(r'^[ACGT]+$')


def parse_info(s):
    out = {}
    for kv in s.split(';'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            out[k] = v
    return out


def read_vcf(path):
    """Yield (chrom, pos, id, ref, alt, info) for every record."""
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            c = line.rstrip('\n').split('\t')
            yield c[0], int(c[1]), c[2], c[3], c[4], parse_info(c[7])


def vcf_date(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('##fileDate'):
                return line.strip().split('=', 1)[1]
            if not line.startswith('##'):
                return None


def label_of(info):
    sig = info.get('CLNSIG', '')
    if sig in PATH_LABELS:
        return 1
    if sig in BEN_LABELS:
        return 0
    return None


def main():
    counts = {}
    cur_path = os.path.join(DATA, 'clinvar.vcf.gz')
    old_path = os.path.join(DATA, 'clinvar_20230107.vcf.gz')
    counts['clinvar_release'] = vcf_date(cur_path)
    counts['clinvar_temporal_reference_release'] = vcf_date(old_path)

    # ---- 1. filter current release ----
    rows, n_total = [], 0
    drop = dict(label=0, stars=0, chrom=0, type=0, allele=0, size=0, gene=0)
    for chrom, pos, vid, ref, alt, info in read_vcf(cur_path):
        n_total += 1
        y = label_of(info)
        if y is None:
            drop['label'] += 1; continue
        if info.get('CLNREVSTAT', '') not in STAR1:
            drop['stars'] += 1; continue
        if chrom not in CHROMS:
            drop['chrom'] += 1; continue
        if info.get('CLNVC', '') not in TYPES:
            drop['type'] += 1; continue
        if not (ACGT.match(ref) and ACGT.match(alt)):
            drop['allele'] += 1; continue
        if abs(len(ref) - len(alt)) > MAX_INDEL or max(len(ref), len(alt)) > MAX_INDEL + 1:
            drop['size'] += 1; continue
        gi = info.get('GENEINFO')
        if not gi:
            drop['gene'] += 1; continue
        gene = gi.split('|')[0].split(':')[0]
        mc = info.get('MC', '')
        mc = mc.split(',')[0].split('|')[1] if '|' in mc else 'unknown'
        rows.append((vid, chrom, pos, ref, alt, y, gene, info['CLNVC'], mc, info['CLNSIG'], info['CLNREVSTAT']))
    df = pd.DataFrame(rows, columns=['id', 'chrom', 'pos', 'ref', 'alt', 'label', 'gene', 'vtype', 'consequence',
                                     'clnsig', 'revstat'])
    df = df.drop_duplicates(['chrom', 'pos', 'ref', 'alt']).reset_index(drop=True)
    counts['vcf_records_total'] = n_total
    counts['dropped'] = drop
    counts['eligible'] = {'total': len(df), 'pathogenic': int(df.label.sum()), 'benign': int((df.label == 0).sum())}
    print('eligible', counts['eligible'])

    # ---- 2. balanced random sample ----
    rng = np.random.default_rng(SEED)
    parts = []
    for y in (0, 1):
        sub = df[df.label == y]
        n = min(N_PER_CLASS, len(sub))
        parts.append(sub.iloc[rng.choice(len(sub), n, replace=False)])
    df = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ---- 3. sequence windows from hg38 ----
    tb = twobitreader.TwoBitFile(os.path.join(DATA, 'hg38.2bit'))
    idx = {b: i for i, b in enumerate('ACGT')}
    X = np.zeros((len(df), 8, W), dtype=np.uint8)   # rows 0-3: reference window, rows 4-7: alternate window
    keep = np.ones(len(df), dtype=bool)
    ref_mismatch = 0
    for i, r in enumerate(df.itertuples()):
        seq = tb['chr' + r.chrom]
        p0 = r.pos - 1                                   # 0-based start of REF
        left = seq[p0 - FLANK:p0].upper()
        refseq = seq[p0:p0 + len(r.ref)].upper()
        right = seq[p0 + len(r.ref):p0 + len(r.ref) + W].upper()
        if refseq != r.ref or len(left) != FLANK:
            keep[i] = False; ref_mismatch += 1; continue
        ref_win = (left + seq[p0:p0 + W].upper())[:W]
        alt_win = (left + r.alt + right)[:W]
        for j, b in enumerate(ref_win):
            if b in idx: X[i, idx[b], j] = 1
        for j, b in enumerate(alt_win):
            if b in idx: X[i, 4 + idx[b], j] = 1
    df, X = df[keep].reset_index(drop=True), X[keep]
    counts['ref_allele_mismatch_dropped'] = ref_mismatch
    counts['final'] = {'total': len(df), 'pathogenic': int(df.label.sum()), 'benign': int((df.label == 0).sum()),
                       'genes': int(df.gene.nunique())}
    print('final', counts['final'])

    # ---- 4a. gene-disjoint split 70/15/15 (genes shuffled, assigned greedily by variant count) ----
    genes = df.gene.value_counts()
    order = rng.permutation(genes.index.values)
    target = {'train': 0.70 * len(df), 'val': 0.15 * len(df)}
    assign, filled = {}, {'train': 0, 'val': 0, 'test': 0}
    for g in order:
        k = 'train' if filled['train'] < target['train'] else ('val' if filled['val'] < target['val'] else 'test')
        assign[g] = k
        filled[k] += genes[g]
    df['split_gene'] = df.gene.map(assign)

    # ---- 4b. random variant-level split 70/15/15, stratified (for the leakage comparison) ----
    from sklearn.model_selection import train_test_split
    tr, rest = train_test_split(np.arange(len(df)), test_size=0.30, stratify=df.label, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=df.label.values[rest], random_state=SEED)
    df['split_random'] = ''
    df.loc[tr, 'split_random'] = 'train'; df.loc[va, 'split_random'] = 'val'; df.loc[te, 'split_random'] = 'test'

    # ---- 5. temporal flag: variant had no qualifying P/LP or B/LB (>=1 star) record in the Jan-2023 release ----
    old = {}
    for chrom, pos, vid, ref, alt, info in read_vcf(old_path):
        if label_of(info) is not None and info.get('CLNREVSTAT', '') in STAR1:
            old[vid] = 1
    df['new_since_2023'] = ~df.id.isin(old.keys())

    # ---- split summaries ----
    for s in ('split_gene', 'split_random'):
        counts[s] = {}
        for k in ('train', 'val', 'test'):
            sub = df[df[s] == k]
            counts[s][k] = {'n': len(sub), 'pathogenic': int(sub.label.sum()), 'benign': int((sub.label == 0).sum()),
                            'genes': int(sub.gene.nunique())}
    gt = df[df.split_gene == 'test']
    counts['gene_split_gene_overlap'] = {
        'train_test': len(set(df[df.split_gene == 'train'].gene) & set(gt.gene)),
        'train_val': len(set(df[df.split_gene == 'train'].gene) & set(df[df.split_gene == 'val'].gene))}
    rt = df[df.split_random == 'test']
    counts['random_split_test_variants_whose_gene_is_in_train'] = float(
        rt.gene.isin(set(df[df.split_random == 'train'].gene)).mean())
    counts['temporal_subset_of_gene_test'] = {'n': int(gt.new_since_2023.sum()),
                                              'pathogenic': int(gt[gt.new_since_2023].label.sum()),
                                              'benign': int((gt[gt.new_since_2023].label == 0).sum())}
    counts['vtype_by_label'] = {k: {str(a): int(b) for a, b in v.items()}
                                for k, v in df.groupby('vtype').label.value_counts().unstack(fill_value=0).to_dict('index').items()}
    counts['params'] = dict(seed=SEED, flank=FLANK, max_indel=MAX_INDEL, n_per_class=N_PER_CLASS)

    np.savez_compressed(os.path.join(OUT, 'dataset.npz'), X=X, y=df.label.values.astype(np.uint8))
    df.to_csv(os.path.join(OUT, 'variants.csv'), index=False)
    with open(os.path.join(OUT, 'prep_summary.json'), 'w') as f:
        json.dump(counts, f, indent=2)
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
