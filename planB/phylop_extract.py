"""Round 2 (R2-3): phyloP100way score at each of the 101 window positions (reference coordinates).

Usage: python phylop_extract.py
Output: out/full/phylop_full.npy  (n, 101) float32, NaN -> 0; out/full/phylop_summary.json
"""
import json, os
import numpy as np
import pandas as pd
import pybigtools

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, 'out', 'full')
BW = os.path.join(HERE, 'data', 'hg38.phyloP100way.bw')


def main():
    df = pd.read_csv(os.path.join(FULL, 'variants_full.csv'), dtype={'chrom': str})
    bw = pybigtools.open(BW)
    out = np.zeros((len(df), 101), dtype=np.float32)
    order = np.lexsort((df.pos.values, df.chrom.values))
    n_nan = 0
    for k, i in enumerate(order):
        c, p = 'chr' + df.chrom.values[i], int(df.pos.values[i])
        v = np.asarray(bw.values(c, p - 1 - 50, p - 1 + 51), dtype=np.float32)
        nan = np.isnan(v)
        n_nan += int(nan.sum())
        v[nan] = 0.0
        out[i] = v
        if k % 50000 == 0:
            print(k, flush=True)
    np.save(os.path.join(FULL, 'phylop_full.npy'), out)
    json.dump({'source': 'UCSC hg38.phyloP100way.bw', 'missing_fraction': n_nan / out.size,
               'mean_at_variant_by_label': {str(l): float(out[df.label.values == l, 50].mean()) for l in (0, 1)}},
              open(os.path.join(FULL, 'phylop_summary.json'), 'w'), indent=2)
    print('done')


if __name__ == '__main__':
    main()
