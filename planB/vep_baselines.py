"""Plan B, step 3: score the gene-disjoint test variants with SIFT, PolyPhen-2 and CADD via the Ensembl VEP REST API.

Score selection per variant: the canonical transcript's SIFT / PolyPhen-2 score if present, otherwise the most
damaging score over all transcripts (lowest SIFT, highest PolyPhen-2). CADD PHRED is variant-level.
Responses are cached in out/vep_cache.jsonl so the script can be re-run after interruption.

Usage: python vep_baselines.py
Output: out/vep_scores.csv, out/vep_info.json
"""
import json, os, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
CACHE = os.path.join(OUT, 'vep_cache.jsonl')
URL = 'https://rest.ensembl.org/vep/homo_sapiens/region?CADD=1&canonical=1'
BATCH = 200


def post(variants, tries=3):
    """POST a batch; if it keeps failing, split it in half so one bad variant cannot block the rest.
    A single variant that still fails is returned as an empty record (counted as 'not scored')."""
    body = json.dumps({'variants': variants}).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json',
                                                                  'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception as e:
            wait = 5 * 2 ** attempt
            print('retry', attempt, len(variants), e, 'sleep', wait, flush=True)
            time.sleep(wait)
    if len(variants) == 1:
        print('vep_failed_variant', variants[0], flush=True)
        return [{'input': variants[0], 'vep_failed': True}]
    mid = len(variants) // 2
    return post(variants[:mid]) + post(variants[mid:])


def pick(v):
    tcs = v.get('transcript_consequences', [])
    canon = [t for t in tcs if t.get('canonical')]
    def score(key, best):
        for group in (canon, tcs):
            vals = [t[key] for t in group if t.get(key) is not None]
            if vals:
                return best(vals)
        return None
    cadd = None
    for t in tcs + v.get('intergenic_consequences', []) + v.get('regulatory_feature_consequences', []):
        if t.get('cadd_phred') is not None:
            cadd = t['cadd_phred']; break
    return dict(sift=score('sift_score', min), polyphen=score('polyphen_score', max), cadd_phred=cadd,
                vep_consequence=v.get('most_severe_consequence'))


def main():
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'test'
    meta = pd.read_csv(os.path.join(OUT, 'variants.csv'), dtype={'chrom': str})
    test = meta[meta.split_gene == split]
    key = lambda r: f"{r.chrom} {r.pos} {r.id} {r.ref} {r.alt} . . ."
    todo = {key(r): r.id for r in test.itertuples()}
    done = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            rec = json.loads(line)
            done[rec['input']] = rec
    pending = [k for k in todo if k not in done]
    print(f'{len(todo)} test variants, {len(pending)} to query', flush=True)
    with open(CACHE, 'a') as f:
        for i in range(0, len(pending), BATCH):
            res = post(pending[i:i + BATCH])
            for v in res:
                rec = dict(input=v['input'], **pick(v), vep_failed=bool(v.get('vep_failed')))
                done[rec['input']] = rec
                f.write(json.dumps(rec) + '\n')
            f.flush()
            print(f'{min(i + BATCH, len(pending))}/{len(pending)}', flush=True)
            time.sleep(1)
    rows = []
    for k, vid in todo.items():
        rec = done.get(k, {})
        rows.append(dict(id=vid, sift=rec.get('sift'), polyphen=rec.get('polyphen'),
                         cadd_phred=rec.get('cadd_phred'), vep_consequence=rec.get('vep_consequence'),
                         vep_returned=k in done and not rec.get('vep_failed')))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'vep_scores.csv' if split == 'test' else f'vep_scores_{split}.csv'),
                              index=False)
    info = json.load(urllib.request.urlopen(urllib.request.Request(
        'https://rest.ensembl.org/info/data?', headers={'Accept': 'application/json'})))
    sw = json.load(urllib.request.urlopen(urllib.request.Request(
        'https://rest.ensembl.org/info/software?', headers={'Accept': 'application/json'})))
    json.dump({'ensembl_data_release': info, 'ensembl_software': sw, 'endpoint': URL,
               'queried_on': time.strftime('%Y-%m-%d')}, open(os.path.join(OUT, 'vep_info.json'), 'w'), indent=2)
    print('done')


if __name__ == '__main__':
    main()
