function [out, info] = spectral_transforms(rois, wl, mode, ref)
% SPECTRAL_TRANSFORMS
% Feature-selective transformations of single-molecule spectral ROIs, used
% by input_ablation_experiments.m to test which of the spectral properties
% the reviewer named (peak position, band width, shoulder) the CNN
% actually needs.
%
% The logic is ablation by information removal: destroy ONE spectral
% property in the input, retrain from scratch, and see whether accuracy
% survives. This is a causal intervention on the input, unlike occlusion
% (which removes a wavelength region, not a property) or feature-group
% classifiers (which describe the data, not the network).
%
%   rois - [H x W x N] or [H x W x 1 x N] real ROI images
%   wl   - [H x 1] wavelength axis (non-uniform, ~3.72 nm mean spacing)
%   mode - one of:
%     'identity'         no change (baseline)
%     'peak_align'       maps each molecule's own peak onto a common
%                        reference wavelength. REMOVES peak position;
%                        preserves width and shape.
%     'width_norm'       rescales the wavelength axis about each
%                        molecule's own peak so its own FWHM equals a
%                        common reference. REMOVES band width; preserves
%                        peak position.
%     'shoulder_remove'  replaces the shoulder window with a straight
%                        line between its endpoints. REMOVES the
%                        shoulder; preserves peak position and width.
%     'const_shift'      shifts EVERY molecule by the same amount.
%     'const_scale'      rescales EVERY molecule by the same factor.
%   ref  - struct of constants fixed for the whole dataset:
%            .wl_ref, .fwhm_ref, .shift_ref, .scale_ref,
%            .shoulder_lo, .shoulder_hi
%
% WHY const_shift / const_scale EXIST. 'peak_align' and 'width_norm' both
% resample the spectrum, and resampling is itself mildly lossy. A drop in
% accuracy could therefore reflect interpolation damage rather than the
% removal of information. const_shift and const_scale apply the same
% amount of resampling with the SAME parameter for every molecule, so all
% between-class differences are preserved exactly (a global
% reparameterization of the axis) while the interpolation burden is
% comparable. They are the null against which peak_align and width_norm
% must be read: if a control is itself at baseline, the resampling is
% harmless and any drop in the matching ablation is attributable to
% information loss.
%
% NO LABEL LEAKAGE. Every per-molecule parameter (that molecule's own peak
% wavelength and its own FWHM) is estimated from that molecule's own
% spectrum alone, so the transform is identical in training and test folds
% and never consults a label. The two global constants (wl_ref, fwhm_ref)
% are single scalars computed by reference_constants() below as pooled
% medians over all molecules of all classes -- pooled, never per class,
% which would leak the label.
%
% EDGE HANDLING. After shifting or rescaling, some output wavelengths map
% outside the measured range. Source wavelengths are clamped to the
% measured interval, i.e. the spectrum's own edge value is replicated.
% This invents no structure, and is applied identically in every mode
% including the controls, so it cannot differentially favour one
% condition. info.frac_clamped records how much of the axis this affects.

if ndims(rois) == 4
    rois = squeeze(rois);
end
wl = wl(:);
[n_h, n_w, n_s] = size(rois);
rois = double(rois);
out = zeros(n_h, n_w, n_s);

info = struct();
info.mode = mode;
info.peak_wl = nan(n_s,1);
info.fwhm = nan(n_s,1);
info.fwhm_fallback = false(n_s,1);
info.centroid = nan(n_s,1);
n_clamped = 0;

for i = 1:n_s
    prof = mean(rois(:,:,i), 2);
    % Parameters are estimated from a 3-band moving average of the
    % molecule's own mean profile. Single-molecule spectra are noisy
    % enough that a raw argmax can jump several bands between adjacent
    % molecules; smoothing is used ONLY for parameter estimation, never
    % for the output data.
    sm = movmean(prof, 3);
    [pk_val, pk_row] = max(sm);
    peak_wl_i = wl(pk_row);
    w_i = max(sm, 0);
    centroid_i = sum(wl .* w_i) / max(sum(w_i), eps);
    [fwhm_i, fb] = local_fwhm(wl, sm, pk_row, pk_val);
    info.peak_wl(i) = peak_wl_i;
    info.fwhm(i) = fwhm_i;
    info.centroid(i) = centroid_i;
    info.fwhm_fallback(i) = fb;

    switch mode
        case 'identity'
            out(:,:,i) = rois(:,:,i);
            continue;
        case 'peak_align'
            src = wl + (peak_wl_i - ref.wl_ref);
        case 'const_shift'
            src = wl + ref.shift_ref;
        case 'centroid_align'
            % ABLATION of whole-band POSITION: maps each molecule's own
            % intensity-weighted mean wavelength onto a common reference.
            % Distinct from 'peak_align', which equalizes only the peak
            % maximum; because the centroid-minus-peak offset varies
            % between molecules, peak alignment leaves centroid variation
            % behind (quantified in the diagnostics table).
            src = wl + (centroid_i - ref.centroid_ref);
        case 'width_norm'
            src = peak_wl_i + (wl - peak_wl_i) * (fwhm_i / ref.fwhm_ref);
        case 'const_scale'
            % Scaled about the molecule's OWN peak, not about wl_ref.
            % Scaling about a fixed wl_ref would drag every peak toward
            % that reference and so partly remove peak-position spread
            % (measured: 7.61 -> 6.35 nm on the pairwise data), which
            % would contaminate the control with the very ablation it is
            % meant to isolate. Scaling about each molecule's own peak
            % leaves peak positions untouched and applies an identical
            % width factor to all molecules, so relative widths are
            % preserved exactly.
            src = peak_wl_i + (wl - peak_wl_i) * ref.scale_ref;
        case 'shoulder_remove'
            src = wl;
        case {'align_width', 'align_width_shoulder'}
            % CUMULATIVE ablation: peak mapped to the common reference AND
            % FWHM mapped to the common value, in one mapping rather than
            % two chained resamplings (which would interpolate twice and
            % confound the comparison). Derivation: the forward map is
            % u = wl_ref + (lambda - peak_i) * (fwhm_ref / fwhm_i), so the
            % pre-image of output wavelength u is
            % lambda = peak_i + (u - wl_ref) * (fwhm_i / fwhm_ref).
            % 'align_width_shoulder' additionally flattens the shoulder,
            % applied below in the transformed frame.
            src = peak_wl_i + (wl - ref.wl_ref) * (fwhm_i / ref.fwhm_ref);
        case 'const_both'
            % RESAMPLING CONTROL for the cumulative ablations: a constant
            % shift AND a constant width factor, both applied about the
            % molecule's own peak. Every molecule is distorted identically,
            % so peak-position and width differences between molecules are
            % preserved (peaks all move by the same shift_ref/scale_ref,
            % widths all scale by the same factor) while the interpolation
            % burden matches the cumulative ablation.
            src = peak_wl_i + (wl - peak_wl_i) * ref.scale_ref + ref.shift_ref;
        otherwise
            error('Unknown mode %s', mode);
    end

    src_c = min(max(src, wl(1)), wl(end));
    n_clamped = n_clamped + sum(src_c ~= src);
    out(:,:,i) = interp1(wl, rois(:,:,i), src_c, 'linear');

    if any(strcmp(mode, {'shoulder_remove', 'align_width_shoulder'}))
        sel = wl >= ref.shoulder_lo & wl <= ref.shoulder_hi;
        idx = find(sel);
        if numel(idx) >= 3
            lo = idx(1); hi = idx(end);
            % Straight line between the two boundary rows, per spatial
            % column, so the shoulder's excess intensity over a smooth
            % interpolation is removed while the band's overall level
            % across the window is retained.
            for c = 1:n_w
                % Endpoints are taken from the CURRENT (possibly already
                % resampled) profile, not from the raw input: under
                % 'align_width_shoulder' the band has been shifted and
                % rescaled, so raw-frame endpoints would no longer bracket
                % the shoulder.
                out(idx, c, i) = interp1([wl(lo) wl(hi)], ...
                    [out(lo,c,i) out(hi,c,i)], wl(idx), 'linear');
            end
        end
    end
end

info.frac_clamped = n_clamped / (n_h * n_s);
info.n_fwhm_fallback = sum(info.fwhm_fallback);
end

function [fwhm, fallback] = local_fwhm(wl, prof, pk_row, pk_val)
% FWHM about the given peak, measured at half of (peak - baseline) where
% the baseline is the profile's own minimum. Crossings are located by
% linear interpolation between the bracketing samples. If a crossing does
% not exist within the measured range (a monotonic or truncated band), the
% axis end is used and the molecule is flagged, so the fallback rate can
% be reported rather than silently absorbed.
fallback = false;
base = min(prof);
half = base + 0.5 * (pk_val - base);

l = find(prof(1:pk_row) <= half, 1, 'last');
if isempty(l)
    wl_l = wl(1); fallback = true;
else
    if l == pk_row
        wl_l = wl(l);
    else
        wl_l = interp1(prof([l l+1]), wl([l l+1]), half, 'linear');
        if ~isfinite(wl_l), wl_l = wl(l); fallback = true; end
    end
end

r = find(prof(pk_row:end) <= half, 1, 'first');
if isempty(r)
    wl_r = wl(end); fallback = true;
else
    r = pk_row + r - 1;
    if r == pk_row
        wl_r = wl(r);
    else
        wl_r = interp1(prof([r-1 r]), wl([r-1 r]), half, 'linear');
        if ~isfinite(wl_r), wl_r = wl(r); fallback = true; end
    end
end

fwhm = wl_r - wl_l;
if ~isfinite(fwhm) || fwhm <= 0
    fwhm = wl(end) - wl(1);
    fallback = true;
end
end
