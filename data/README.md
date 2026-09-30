# Data format

Each `processed_combined/bodipy{N}.mat` file contains every detected single-molecule spectral ROI for one BODIPY dye variant (N = 1-7, 9-15; dye 8 was not fluorescent and has no file). All molecules from every acquisition session for that dye have already been pooled into these arrays.

## Fields

| Field | Size | Description |
|---|---|---|
| `wl` | `[1 x 55]` | Wavelength (nm) of each of the 55 spectral bins, common to every dye. Spans 599.7-800.7 nm. |
| `spectral_rois_combined` | `[55 x 5 x N]` | The raw 2D spectral ROI for each of the N detected molecules: 55 wavelength bins (rows) x 5 spatial pixels (columns, the point-spread-function's spatial extent transverse to the spectral dispersion direction). Intensity is min-max normalized to `[0, 1]` per file. This is the direct CNN input (reshaped to `[55 x 5 x 1 x N]`). |
| `spectral_profiles_combined` | `[55 x N]` | The same data collapsed to a 1D spectrum per molecule (sum/mean across the 5 spatial pixels) -- used by the feature-extraction and interpretability scripts, not by the CNN training scripts directly. |
| `n_photons_combined` | `[N x 1]` | Total detected photon count for each molecule. Used for the `> 1000` quality filter applied by every training script. |
| `spectral_centroids_combined` | `[N x 1]` | Intensity-weighted mean wavelength for each molecule. Used for outlier rejection (`rmoutliers`) before training. |
| `peak_profiles_combined` | `[N x 2]` | Peak-fit parameters from the upstream spectral-calibration step (not used by the scripts in this repository). |
| `coeffs_combined` | `[N x 2]` | Additional per-molecule fit coefficients from the upstream pipeline (not used by the scripts in this repository). |

`N` (the number of molecules) varies by dye, from ~1,000 to ~13,000 before any filtering is applied (see the CNN scripts' own `n_photons_combined > 1000` + centroid-outlier filtering + balanced subsampling, described in the top-level README).

## Loading example

```matlab
d = load('data/processed_combined/bodipy1.mat');
sel = d.n_photons_combined > 1000;               % photon-count filter
rois = d.spectral_rois_combined(:,:,sel);         % [55 x 5 x n_selected]
wl = d.wl;                                        % [1 x 55], shared across all dyes
```

## Provenance

These files are the output of an upstream image-processing pipeline (single-molecule localization, spectral ROI extraction, wavelength calibration) that is not part of this repository. They are the same files referenced by the manuscript's Data Availability Statement.
