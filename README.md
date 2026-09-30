# CNN-Based Spectral Classification of BODIPY Dyes

This repository contains the complete machine-learning workflow for the CNN-based single-molecule spectral classification described in the manuscript *"Photoactivatable BODIPYs for Simultaneous Spectral Single-Molecule Discrimination"* — model architecture, training/evaluation scripts, data preprocessing, train/validation/test split information, and step-by-step instructions to reproduce every reported classification metric.

It contains two generations of the analysis, both included because both sets of numbers appear in the manuscript:

- **`src/original_single_split/`** — the original scripts, trained/evaluated on a single fixed data split. These produced the manuscript's originally-reported numbers (e.g. the pairwise accuracy heatmap in Fig. 4B, the 1-6-13 triad in Fig. 4D-E).
- **`src/cross_validation/`** — stratified 5-fold cross-validation of the same architecture and preprocessing, added in response to reviewer comments requesting repeated-split validation with mean ± standard deviation. This is what the revised manuscript reports as the primary robustness evidence.

---

## 1. Repository structure

```
publish/
├── README.md                          <- this file
├── LICENSE                            <- MIT
├── CITATION.cff                       <- machine-readable citation (placeholders -- see Citation section)
├── data/
│   ├── README.md                      <- data format documentation
│   └── processed_combined/
│       └── bodipy{1,2,3,4,5,6,7,9,10,11,12,13,14,15}.mat  <- one file per dye (14 dyes; dye 8 was not fluorescent)
├── src/
│   ├── original_single_split/         <- single 70/15/15-style split (Section 3)
│   │   ├── analyze_bodipy_pairs.m         (all 91 pairwise combinations)
│   │   ├── analyze_bodipy_triplets2.m     (systematic sweep, all 364 triad combinations)
│   │   ├── analyze_bodipy_triplets_1_3_9.m
│   │   ├── analyze_bodipy_triplets_1_4_13.m
│   │   ├── analyze_bodipy_triplets_1_6_13.m   <- the triad reported in Fig. 4D-E
│   │   ├── analyze_bodipy_triplets_2_11_13.m
│   │   ├── analyze_bodipy_triplets_3_5_13.m
│   │   ├── analyze_bodipy_triplets_3_13_15.m
│   │   ├── calculate_metrics.m            (shared accuracy/precision/recall/F1 helper)
│   │   └── formatfig.m                    (minimal figure-export helper -- see note in Section 6)
│   ├── cross_validation/               <- stratified 5-fold CV (Section 4)
│   │   ├── pairwise_cv_bodipy_new_num.m   (all 91 pairs)
│   │   ├── triplet_cv_bodipy_new_num.m    (the same 6 representative triads)
│   │   ├── pairwise_cv_grouped.m          (acquisition-file-grouped CV; needs upstream files NOT shipped -- see Section 5.2)
│   │   └── cv_vs_singlesplit_figure.m     (single-split vs cross-validated scatter, all 91 pairs)
│   ├── interpretability/               <- addresses the reviewer's interpretability comment (Section 5.3)
│   │   ├── spectral_transforms.m          (feature-targeting input transforms)
│   │   ├── reference_constants.m          (their dataset-wide reference constants)
│   │   ├── test_spectral_transforms.m     (synthetic validation; run BEFORE the experiments)
│   │   ├── input_ablation_experiments.m   <- primary evidence
│   │   ├── occlusion_multiseed.m          (occlusion profile across 10 independently trained networks)
│   │   ├── occlusion_band_importance.m    (single-network occlusion profile + controls)
│   │   ├── descriptor_group_comparison.m  (descriptor groups, matched dimensionality)
│   │   ├── features_only_classifier.m     (earlier 3-group version, superseded by the above)
│   │   ├── extract_spectral_features.m    (the 10 explicit spectral descriptors)
│   │   ├── cnn_weight_analysis.m          (first-layer filter analysis -- reported, NOT used as evidence)
│   │   ├── reverify_interpretability.m        (audit of the filter analysis)
│   │   ├── reverify_interpretability_round2.m (random-init control for the same)
│   │   └── test_feature_extraction_demo.m
│   ├── save_figure.m                   <- shared figure-export helper (PNG, and EPS when run)
│   └── smoke_test.m                    <- quick (~1 min) sanity check, see Section 5.0
└── results/
    ├── cross_validation/
    │   ├── pairwise_cv_summary.txt     <- all 91 pairs, mean +/- std
    │   ├── pairwise_cv_folds.csv       <- raw per-fold accuracy/precision/recall/F1 underlying the summary above
    │   ├── triplet_cv_summary.txt      <- all 6 triads, mean +/- std, per-class
    │   ├── triplet_cv_folds.csv        <- raw per-fold metrics underlying the summary above
    │   ├── pairwise_grouped_cv_summary.txt  <- acquisition-file-grouped CV (partial: 2 of 6 attempted pairs resolved)
    │   └── cv_vs_singlesplit.png       <- single-split vs cross-validated accuracy, all 91 pairs (r = 0.94)
    └── interpretability/
        ├── ablation/                   <- input_ablation_summary.txt (+ pooled-reference sensitivity),
        │                                  descriptor_group_summary.txt, and the input_ablation_* /
        │                                  descriptor_groups_* figures
        ├── multiseed/                  <- occlusion_multiseed_summary.txt and the four
        │                                  occlusion_multiseed_* figures
        ├── occlusion/                  <- occlusion_band_importance_summary.txt + per-window CSVs,
        │                                  features_only_classifier_summary.txt, and the
        │                                  occlusion_* / feature_group_* figures
        ├── reverification/             <- reverification_findings.txt (curated), the generated
        │                                  summaries, peak tables, and filter-response figures
        ├── pairwise/                   <- cnn_weight_analysis output for the BODIPY-4/10 network
        │                                  (filter, activation and saliency figures + summary)
        └── triplet/                    <- the same for the BODIPY-2/11/13 network
```

Running the scripts in `src/` creates additional subfolders under `results/` (e.g. `results/original_single_split/`, and the `.mat` checkpoints/reference nets in `results/cross_validation/`) — those are not shipped, only generated when you run something.

What **is** shipped under `results/` is every summary table, per-fold/per-window CSV, and the figures for the cross-validation and interpretability analyses as `.png`, so the reported numbers and figures can be checked without first reproducing a multi-hour run. Vector `.eps` versions are not shipped; re-running any figure script writes both formats (Section 5.4). Figures are named for the analysis that produced them rather than for their position in the manuscript response, so a figure can be traced directly back to the script and summary it came from.

All scripts resolve the data directory, and by default their own output directory, relative to their own file location (`repo_root/data/processed_combined`, `repo_root/results/...`) rather than to MATLAB's current working directory — so they produce the same output regardless of where you invoke them from. The one exception: if you explicitly override `cnn_weight_analysis`'s `'output_dir'` argument with a *relative* path (as the examples in Section 5.3 do), that path is resolved against your current directory, same as any MATLAB function — pass an absolute path instead if you're not running from the repository root.

---

## 2. Requirements

- MATLAB R2021b or later (tested on R2025a)
- Deep Learning Toolbox (`trainNetwork`, `dlnetwork`, `dlgradient`, `occlusionSensitivity`)
- Statistics and Machine Learning Toolbox (`cvpartition`, `rmoutliers`, `anova1`)
- Image Processing Toolbox (`imageDataAugmenter`)
- A CUDA-capable GPU is optional but strongly recommended (training all 91 pairs takes ~2-2.5 hours on a GPU; substantially longer on CPU only). If your GPU's compute capability is newer than what your MATLAB version's bundled CUDA libraries natively support, call `parallel.gpu.enableCUDAForwardCompatibility(true)` before training (the cross-validation scripts already do this automatically, with a CPU fallback if no GPU is present).

---

## 3. Data

See [`data/README.md`](data/README.md) for the exact file format (fields, dimensions, units). In brief: `data/processed_combined/bodipy{N}.mat` contains every detected single-molecule spectral ROI for dye N, already extracted from the raw camera frames and interpolated onto a common 55-point wavelength grid (599.7-800.7 nm) — this is the preprocessed representation the CNNs are trained on directly. Preprocessing applied identically by every script in this repository before training:

1. **Photon-count filtering**: molecules with `n_photons_combined <= 1000` are discarded (rejects dim, high-uncertainty spectra).
2. **Centroid outlier removal**: `rmoutliers(spectral_centroids, 'mean', 'thresholdfactor', 4)` removes molecules whose spectral centroid is a statistical outlier.
3. **Balanced subsampling**: a fixed number of molecules per dye is drawn at random (without replacement) so every class contributes equally — 600/dye for pairwise classification; 500-700/dye for the triads (see the table in Section 4).
4. **Min-max normalization**: each ROI's intensity is rescaled to `[0, 1]` (this happens upstream, in the data files themselves — `spectral_rois_combined` is already normalized).

**Scope note:** this repository ships the ~63 MB processed dataset above (`data/processed_combined/`, 14 dyes) and everything downstream of it: preprocessing, model training/evaluation, and interpretability analysis. The raw camera images and the upstream extraction pipeline (ROI cropping, spectral calibration, and the conversion from raw frames to `spectral_rois_combined`/`n_photons_combined`/etc.) are intentionally *out of scope* for this repository — they are a separate, much larger acquisition-and-calibration pipeline, not part of the machine-learning workflow this repository documents. The data underlying the study, including these processed files, is available as described in the manuscript's Data Availability Statement.

---

## 4. Model architecture and train/validation/test splits

Both classification tasks (pairwise and triplet) use the same **dual-branch CNN**, taking a `[55 x 5 x 1]` spectral ROI image as input:

```
Input [55 x 5 x 1]
 |
 +-- Branch 1 (spectral shape) --------------------------------------+
 |     Conv2D(3x3, F1 filters, same padding) -> BatchNorm -> ReLU    |
 |     MaxPool(2x2, stride 2)                                        |
 |     Conv2D(3x3, F2 filters, same padding) -> BatchNorm -> ReLU    |
 |     MaxPool(2x2, stride 2)                                        |
 |     Dropout(0.5)                                                  |
 |     FullyConnected(32)                                             |
 |                                                                    |
 +-- Branch 2 (direct/positional) ------------------------------------+
       AveragePool([1 x 5])  (collapses the 5-pixel width dimension)
       FullyConnected(32)

Concatenate(Branch1, Branch2) -> ReLU -> FullyConnected(n_classes) -> Softmax
```

- **Pairwise classification** (`analyze_bodipy_pairs.m` / `pairwise_cv_bodipy_new_num.m`): F1=16, F2=32, `n_classes=2`.
- **Triplet classification** (all `analyze_bodipy_triplets_*.m` / `triplet_cv_bodipy_new_num.m`): F1=16, F2=32, `n_classes=3`.

Training: SGDM optimizer, initial learning rate 0.001, 200 epochs, `RandXReflection` data augmentation on the training set only (doubles effective training set size by flipping each spectral image along its wavelength axis).

### Splits

| Task | Original single split | Cross-validation |
|---|---|---|
| Pairwise (per pair) | 600 molecules/dye total, split 420 train / 90 validation / 90 test (70/15/15) | Same 600/dye pool, stratified 5-fold (folds ~480 train / 120 validation each) |
| Triplet 1-3-9 | 500/dye, 400 train / 100 test | 500/dye, stratified 5-fold |
| Triplet 1-4-13 | 600/dye, 420/90/90 | 600/dye, stratified 5-fold |
| Triplet 1-6-13, 3-5-13, 3-13-15 | 700/dye, 420/90/(remainder~190) | 700/dye, stratified 5-fold |
| Triplet 2-11-13 | 610/dye, 420/90/100 | 610/dye, stratified 5-fold |

All splits are drawn with a fixed random seed (`rng(42, 'twister')` in the CV scripts; `rng(1)`/`rng(2)`/`rng(9)` per triad in the original scripts, matching each one's own header) for reproducibility. Cross-validation folds are stratified (`cvpartition(..., 'Stratify', true)`) to preserve class balance in every fold, and `ValidationData` is used only for training-progress monitoring, never for model selection — there is no leakage between the reported validation-fold metrics and the trained weights.

**A calibration note on exact reproducibility:** the fixed seed makes the *data split* (which molecules go into which fold/set) exactly reproducible on any machine. The *trained weights*, however, are not guaranteed bit-identical across different hardware/GPU driver/cuDNN versions — this is a general property of `trainNetwork`, not specific to this code. In our own testing, rerunning `analyze_bodipy_triplets_1_6_13.m` reproduces its result exactly (down to the confusion matrix) on the *same* machine every time, but on a *different* machine you should expect results close to, not necessarily identical to, the numbers quoted in Section 5 — typically within the fold-to-fold variation already quantified by the cross-validation study (Section 5.2). Triads 1-4-13 and 1-6-13 are the least stable in this respect (see `results/cross_validation/triplet_cv_summary.txt`: BODIPY-4 and BODIPY-6 recall has the largest standard deviation of any class in any triad), so a bigger swing on those two specifically is expected, not a sign of a broken setup.

---

## 5. Reproducing the reported classification metrics

Add the relevant `src/` subfolder(s) to your MATLAB path first:

```matlab
addpath('src');                       % shared helpers (save_figure, smoke_test)
addpath('src/original_single_split');
addpath('src/cross_validation');
addpath('src/interpretability');
```

`addpath('src')` matters for more than the smoke test: `save_figure.m` lives there and is
called by every script that produces a figure.

### 5.0 Quick sanity check (optional, ~1 minute)

Before committing to the multi-hour full run below, `src/smoke_test.m` checks that your MATLAB installation and toolboxes are correctly set up and that the data files load and shape-check correctly, by loading two dye files and running a few epochs of training on a tiny (80-molecule) subset. It runs as a function, so it leaves your MATLAB workspace untouched:

```matlab
addpath('src');
smoke_test    % ~1 min; prints PASS/FAIL for 5 checks. Not a meaningful accuracy result --
              % see Sections 5.1/5.2 below for the real numbers.
```

### 5.1 Original single-split numbers (manuscript Fig. 4B, 4D-E)

```matlab
analyze_bodipy_pairs        % ~15-45 min. Reproduces the full 91-pair accuracy/precision/recall/F1
                             % heatmap (Fig. 4B), including: 54/91 pairs with accuracy > 0.70;
                             % maximum accuracy 0.911 for pair 1-13; pair 2-11 accuracy 0.78.

analyze_bodipy_triplets_1_6_13   % ~2-5 min. Reproduces the Fig. 4D-E triad: accuracy 0.77,
                                  % per-class precision/recall 71%/81%/78% for BODIPY-1/6/13.

% The other 5 individually-scripted triads (1-3-9, 1-4-13, 2-11-13, 3-5-13, 3-13-15)
% run the same way and are the individual results underlying the systematic sweep below.

analyze_bodipy_triplets2    % OPTIONAL, several hours: exhaustive sweep of all 364 possible
                             % triad combinations (Fig. S60-S61), only 1 of 364 combinations
                             % run by default (edit the `for ii = ...` loop to sweep more).
```

Each script prints its accuracy/precision/recall/F1 to the console (`disp(...)`) and additionally saves figures/matrices under `results/original_single_split/` (relative to the repository root, regardless of your current directory) — e.g. `analyze_bodipy_pairs.m` saves the full pairwise matrices to `results/original_single_split/pseudomix/matrices.mat`, and `analyze_bodipy_triplets2.m` saves each swept combination to `results/original_single_split/triplet/analyze_bodipy_triplets_<ii>.mat`.

### 5.2 Cross-validated numbers (revised manuscript, response to reviewers)

```matlab
pairwise_cv_bodipy_new_num   % ~2-2.5 hours on GPU. All 91 pairs, stratified 5-fold CV.
                              % Expect: mean accuracy across all pairs ~0.73 (median ~0.74);
                              % pair 2-11: 0.782 +/- 0.022; pair 1-13: 0.855 +/- 0.019;
                              % pair 6-7 (hardest): 0.503 +/- 0.006 (chance level).
                              % Results: results/cross_validation/pairwise_cv_results.mat,
                              % pairwise_cv_summary.txt (compare against the copy already
                              % included in this repo to verify your run reproduced ours).

triplet_cv_bodipy_new_num    % ~15-20 min on GPU. All 6 representative triads.
                              % Expect: accuracy 0.696-0.749 depending on triad; triad
                              % 1-6-13: 0.704 +/- 0.032 (compare to the 0.77 single-split
                              % value above -- the CV mean is the more robust estimate).
```

Both scripts checkpoint results to disk after every pair/triad, and reload that checkpoint on startup, skipping whatever is already done — so an interrupted run resumes rather than restarting from the first pair/triad (delete `results/cross_validation/*_cv_results.mat` to force a full restart). This is a best-effort resume, not a bit-identical continuation of an uninterrupted run: the fixed RNG seed (`rng(42)`) advances differently once pairs/triads are skipped, so the specific 600-molecule subsample drawn for work *after* the resume point will differ slightly from what an uninterrupted run would have drawn — within the same run-to-run variability already captured by the reported fold-to-fold std, not an extra source of error.

Alongside each `*_cv_summary.txt`, both scripts also write `*_cv_folds.csv` — the raw per-fold accuracy/precision/recall/F1 (and per-class precision/recall) that the summary's mean +/- std were computed from, so those standard deviations can be independently recomputed rather than taken on faith.

**Two further checks in this folder.**

```matlab
cv_vs_singlesplit_figure     % instant. Scatter of single-split against cross-validated accuracy
                              % for all 91 pairs (r = 0.94), with fold-standard-deviation bars.
                              % Requires results/original_single_split/pseudomix/matrices.mat,
                              % i.e. run analyze_bodipy_pairs (Section 5.1) first.
                              % A copy of the output is shipped as
                              % results/cross_validation/cv_vs_singlesplit.png.

pairwise_cv_grouped([1 2; 1 13])   % Acquisition-file-grouped CV: folds are whole imaging
                                    % sessions rather than individual molecules, which tests
                                    % whether per-molecule CV is optimistic through
                                    % same-acquisition correlations.
```

**`pairwise_cv_grouped.m` cannot be run from this repository as shipped.** It needs the
per-acquisition-file records and index spreadsheet from the upstream extraction pipeline
(`data/upstream/`), which is out of scope here for the reasons given in Section 3. The script is
included because the result it produces is cited as a limitation in the manuscript, and its
output is shipped as `results/cross_validation/pairwise_grouped_cv_summary.txt`: for the 2 of 6
attempted pairs where file provenance could be reconstructed, BODIPY-1 vs BODIPY-2 falls from
0.667 +/- 0.018 to 0.482 +/- 0.100 (chance level) under grouping, while BODIPY-1 vs BODIPY-13
degrades only modestly, from 0.855 +/- 0.019 to 0.800 +/- 0.074.

### 5.3 Interpretability analysis

These analyses address the reviewer comment asking which spectral features the CNN uses to
discriminate — peak position, shoulder intensity, or overall band shape. They are not required to
reproduce any classification metric in Sections 5.1/5.2.

All of them need a trained network and its held-out data, saved automatically by the
cross-validation scripts (`results/cross_validation/pairwise_reference_net.mat`,
`triplet_reference_net.mat`), so run Section 5.2 first.

**Primary evidence — feature-targeting input transformations with retraining.** The input is
transformed so as to equalize one spectral property across all molecules (the peak-maximum
wavelength, whole-band position/spectral centroid, FWHM, or the shoulder region), the classifier is
retrained from scratch, and the accuracy loss is measured against a *matching resampling control*
that applies the same distortion identically to every molecule — so interpolation damage is not
mistaken for information loss.

```matlab
test_spectral_transforms          % ~30 s. Validates on synthetic spectra that each transform
                                   % removes the property it targets and preserves the others.
                                   % Run this first: an ablation null result is uninterpretable
                                   % if the transform did not actually work.

input_ablation_experiments        % ~2.5-3 h on GPU (10 conditions x 2 dye sets x 5 folds x 5 runs).
                                   % Writes results/interpretability/ablation/.
```

Two design points are worth knowing before interpreting the output. Inference is on the
**across-run paired difference** between a transformation and its control over five independent
training runs, not on the fold-to-fold spread within one run — folds are not independently trained
conditions, and using their spread as an error term reverses several of the conclusions.
**Holm–Bonferroni correction** is applied within each of two pre-specified families (eight
single-property contrasts; four cumulative ones), and nominal and adjusted p-values are both
reported. Each fold's transform reference constants are computed from that fold's training
molecules only.

**Supporting analyses.**

```matlab
occlusion_multiseed               % ~25 min on GPU. Occlusion profile across 10 independently
                                   % trained networks per case, over four dye sets spanning the
                                   % observed difficulty range -- including BODIPY-6/7, which
                                   % classifies at chance and acts as a trained negative control.

occlusion_band_importance         % ~5 min. Single-network occlusion profile with randomly
                                   % initialized control, permutation importance, and baseline/
                                   % window-width robustness checks.

descriptor_group_comparison       % ~3 min. The 10 explicit descriptors grouped into peak maximum,
                                   % band position, width/shape, tail and shoulder, compared both
                                   % as full groups and at matched dimensionality.
```

**Reported but not used as evidence.** `cnn_weight_analysis.m` visualizes first-layer filters and
their forward-pass activations. It cannot answer the interpretability question, for a structural
rather than a fixable reason: activation magnitude reports where the network *responds*, which is
dominated by where the input is large, and randomly initialized filters cluster in the same
600–625 nm region as trained ones. `reverify_interpretability.m` and
`reverify_interpretability_round2.m` are the audit establishing this; their findings are in
`results/interpretability/reverification/`.

```matlab
cnn_weight_analysis('results/cross_validation/pairwise_reference_net.mat', 'output_dir', 'results/interpretability/pairwise');
cnn_weight_analysis('results/cross_validation/triplet_reference_net.mat', 'output_dir', 'results/interpretability/triplet');
reverify_interpretability
reverify_interpretability_round2
```

Note that `results/interpretability/reverification/reverification_findings.txt` is a
**hand-written** analysis document, whereas `reverification_summary.txt` in the same folder is
generated by the script. Running `reverify_interpretability` overwrites the latter, never the
former.

### 5.4 Figures

Every figure-producing script calls `src/save_figure.m`, which writes a raster `.png` and a vector
`.eps` of the same figure side by side, the EPS rendered with the Painters renderer so it contains
editable paths rather than an embedded bitmap. Only the `.png` versions are shipped in this
repository; run any figure script to regenerate the `.eps` alongside them. If the lab's `formatfig`
utility is on the path it is used for house styling; otherwise the export still succeeds and a
warning notes that styling was not applied (see Section 6).

## 6. Notes on faithfulness to the original code

In preparing this repository we found and fixed two issues in the original lab scripts, neither of which changes any previously reported number (both are documented in code comments at the point of the fix):

1. `analyze_bodipy_triplets2.m` had a stray extra parenthesis in its final `save(...)` call, a pre-existing typo that would prevent the script from running to completion at all (a MATLAB parse error). Fixed by removing the extra parenthesis; the surrounding computation is untouched.
2. Several original scripts call an internal lab figure-formatting utility (`formatfig`) that is not part of any released codebase and was not available to include here. We provide a minimal drop-in replacement (`src/original_single_split/formatfig.m`) that exports the same figures without the original's specific cosmetic styling. This affects only supplementary figure appearance, never a computed classification metric — every `formatfig` call in these scripts happens after the relevant accuracy/precision/recall/F1 has already been computed.

3. The newer figure-producing scripts call `src/save_figure.m` rather than `formatfig` directly. It exports PNG and EPS together and calls `formatfig` with three arguments only, so it works with either the lab utility or the stub above; if neither is on the path it falls back to a direct export and warns. As with item 2, this affects figure appearance only — never a computed metric.

No other computational logic was altered from the scripts that generated the manuscript's originally reported numbers.

---

## Citation

The manuscript this code accompanies is currently under review and not yet formally published. If you use this code before a formal citation is available, please reference it as:

> Yeo, W.-H.; Zheng, Y.; Tomassini, A.; Liu, Y.; Shahid, M. A.; Singh, A. K.; Hayter, C. E.; Gong, X.; Zhang, H. F.; Zhang, Y.; Raymo, F. M. "Photoactivatable BODIPYs for Simultaneous Spectral Single-Molecule Discrimination" (manuscript under review). Code: https://github.com/FOIL-NU/bodipy-classification

A machine-readable [`CITATION.cff`](CITATION.cff) is included with the same author list. It records the article as `status: submitted` and deliberately carries no journal, year or DOI, since the manuscript is still under review and no correct value for those fields exists yet — a placeholder there would be ingested by reference managers as though it were real.

<!-- WHEN THE ARTICLE IS PUBLISHED
     Add journal, year, volume/pages and doi to the preferred-citation block in
     CITATION.cff, remove its status: submitted line, and update the citation
     line above. Until then those fields are intentionally absent rather than
     filled with placeholders. -->

## License

MIT — see [LICENSE](LICENSE).
