# Fair 2D vs 1D comparison protocol (fixed before any tuning run, 2026-10-18)

Motivation: the first 2D design pools 2x2 over the 8 input rows, which mixes the reference and
alternate one-hot rows after layer 1, while the 1D design never does. To compare the encodings
fairly, both get the same number of candidate designs and the same selection rule.

Candidates (identical filters 32/64/128, identical dense head 256-dropout0.5-64-sigmoid,
identical training: Adam 1e-3, batch 128, max 40 epochs, early stopping patience 5 on val loss):

2D (Conv2D):
- 2D-A  8x101x1 input, 3x3 kernels, 2x2 pooling                        (original design)
- 2D-B  8x101x1 input, first kernel spans all 8 rows (8x5, valid in height), then 1x3, 1x2 pooling
- 2D-C  2x101x4 input (row 1 = reference, row 2 = alternate; channels = A,C,G,T), 2x3 kernels same-padded, 1x2 pooling
- 2D-D  3x101x4 input (reference, alternate, alternate minus reference), 3x3 kernels same-padded, 1x2 pooling

1D (Conv1D, 101 positions):
- 1D-A  8 channels (ref+alt one-hot), kernel 3                           (original design)
- 1D-B  8 channels, kernel 5 in all layers
- 1D-C  12 channels (ref, alt, alt minus ref), kernel 3
- 1D-D  8 channels, kernel 7 in first layer, 3 afterwards

Selection rule: train every candidate once (seed 0) on the gene-disjoint training set; choose the
single best 2D and the single best 1D candidate by **validation ROC-AUC only**. The test set is not
looked at during selection.

Final evaluation: retrain the two selected designs with seeds 0, 1, 2; evaluate each once on the
gene-disjoint test set; report mean +/- SD across seeds and a paired bootstrap test of the
seed-0 AUC difference on identical test variants. All candidates' validation scores are reported.
Results are reported whichever encoding wins.

# Round 2 protocol (fixed before any round-2 run, 2026-10-18, evening)

Goal: test three pre-specified improvements. Validation and test sets are unchanged from round 1
(same 12,023 validation and 11,952 test variants, same gene assignment), so all results stay comparable
and the test set is still evaluated once per final model.

R2-1 More training data: training set expanded to all eligible ClinVar variants (same filters) whose gene
     is a round-1 training gene or a gene absent from round 1; variants of validation/test genes that are not
     already in the validation/test sets are excluded. All pathogenic variants are kept and benign variants
     are randomly sampled to the same number. Model: 2D-D, same hyper-parameters, seeds 0-2.
R2-3 Conservation: phyloP (hg38 phyloP100way, UCSC) at each of the 101 window positions, added as a second
     input branch (Conv1D 32/64/128 over the 101 x 1 phyloP track, pooled and flattened) concatenated with the
     2D-D trunk before the dense head. Missing phyloP values are set to 0. Trained on the R2-1 training set,
     seeds 0-2. This model is no longer sequence-only and is reported as a hybrid.
R2-2 Complementarity with CADD: logistic regression fitted on the VALIDATION set only, with inputs
     (a) CADD PHRED alone, and (b) CADD PHRED + CNN logit; a missing-CADD indicator is included in both and
     missing PHRED is set to 0. Evaluated once on the test set, on all test variants and on the
     CADD-scored subset, with paired bootstrap tests of (b) - (a). Caveat recorded in advance: the CNN's
     validation predictions come from a model early-stopped on that set, which can make (b) slightly optimistic.

Primary-model selection: among round-1 2D-D, R2-1 and R2-3, the model with the highest mean validation AUC
over seeds 0-2 becomes the reported primary model. All three are reported regardless.
