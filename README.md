# GeneSpectra

Pathogenicity prediction for human SNVs and small indels with an interpretable CNN, plus a web interface.
This repository accompanies the DABCon 2026 paper *"Benchmarking an Interpretable CNN Against Classical
Predictors for Large-Scale Genomic Variant Pathogenicity Classification"* (Kaakarla).

## What the model is
- Input: a 101 bp GRCh38 window around the variant, encoded as a 3×101×4 "image" (reference one-hot,
  alternate one-hot, alternate − reference), plus the UCSC phyloP100way conservation score at each of the
  101 positions.
- Architecture: three Conv2D blocks (32/64/128 filters) on the sequence, three Conv1D blocks on phyloP,
  FC 256 → FC 64 → sigmoid (1,715,553 parameters). The served model averages three training seeds.
- Training data: ClinVar release 2026-10-11, review status ≥ 1 star, P/LP vs B/LB, SNVs and indels ≤ 10 bp,
  split **by gene** (no gene shared between training and validation/test).
- Held-out test set (11,952 variants, 998 unseen genes): ROC-AUC 0.991, F1 0.957, MCC 0.912.
  Baseline comparison, leakage, temporal and ablation results are in `planB/results/`.

## Run the app
```bash
pip install -r requirements.txt
python backend/app.py                 # Flask API on http://127.0.0.1:5000
cd frontend/UI && npm ci && npm run dev   # UI on http://localhost:3000
```
In the input box, enter a **GRCh38 variant** as `chrom:pos:ref:alt` (also accepted: `chr17-7675088-C-T`,
`17 7675088 C T`), for example `17:7675088:C:T` (TP53 R175H). The backend fetches the sequence window and
phyloP scores from the UCSC Genome Browser API (internet required), checks the reference allele, runs the
three models, and returns the prediction, Grad-CAM panels, and the held-out test metrics. A bare DNA sequence
is rejected because the model scores a specific variant.

Note: the UCSC web API's phyloP values can differ from the downloaded bigWig used in training by up to ~0.08,
which changes predictions by at most ~0.005 (checked on held-out test variants). GeneSpectra is a research
prototype and is not validated for clinical use.

## Reproduce the experiments (`planB/`)
`planB/` contains every script used for the paper; `TUNING_PROTOCOL.md` records the model-selection rules,
written before the runs. Inputs (not in the repo, ~11 GB): `planB/data/clinvar.vcf.gz`,
`clinvar_20230107.vcf.gz` (NCBI ClinVar FTP), `hg38.2bit` and `hg38.phyloP100way.bw` (UCSC).
```bash
cd planB
python prep_data.py              # round-1 dataset and gene-disjoint / random splits
python train_models.py 3         # round-1 CNNs (2D, 1D, random split)
python tune_encodings.py         # fair 2D vs 1D comparison (validation-only selection)
python vep_baselines.py test     # SIFT / PolyPhen-2 / CADD via Ensembl VEP
python vep_baselines.py val
python evaluate.py
python prep_full.py && python phylop_extract.py
python train_full.py full_2dD 3 && python train_full.py full_2dD_phylop 3
python evaluate_round2.py && python make_figures_round2.py
```
Outputs are written to `planB/out/`; the result files used in the paper are copied to `planB/results/` and
the figures to `planB/figures/`.

## Legacy files
`model/mutation_prediction_model.h5`, `model/cnn_model.py`, `scripts/` and `data/images/` are from the
earlier prototype (trained on generated dummy sequences) and are not used by the current app or paper.
