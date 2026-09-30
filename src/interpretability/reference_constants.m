function ref = reference_constants(rois, wl, shoulder_lo, shoulder_hi)
% REFERENCE_CONSTANTS
% The dataset-wide constants used by spectral_transforms.m.
%
% wl_ref and fwhm_ref are POOLED medians over all molecules of all classes
% -- pooled, never per class, since a per-class reference would encode the
% label into the transform and manufacture the very information the
% ablation is meant to destroy.
%
% shift_ref and scale_ref set the magnitude of the const_shift and
% const_scale resampling controls. Each is the median absolute per-molecule
% distortion of the corresponding ablation, so a control resamples by a
% typical amount for that ablation while leaving all between-class
% differences intact.

if ndims(rois) == 4
    rois = squeeze(rois);
end
wl = wl(:);
n_s = size(rois, 3);
pk = nan(n_s,1);
fw = nan(n_s,1);
ct = nan(n_s,1);
for i = 1:n_s
    prof = movmean(mean(double(rois(:,:,i)), 2), 3);
    [pv, pr] = max(prof);
    pk(i) = wl(pr);
    fw(i) = fwhm_of(wl, prof, pr, pv);
    % Intensity-weighted mean wavelength: a whole-band POSITION statistic,
    % and the most predictive single descriptor in the data. Aligning the
    % peak maximum does not necessarily equalize it, because the
    % centroid-minus-peak offset varies between molecules -- hence a
    % separate centroid reference and a separate ablation.
    w = max(prof, 0);
    ct(i) = sum(wl .* w) / max(sum(w), eps);
end

ref = struct();
ref.wl_ref = median(pk);
ref.centroid_ref = median(ct);
ref.fwhm_ref = median(fw);
ref.shift_ref = median(abs(pk - ref.wl_ref));
ref.scale_ref = 1 + median(abs(fw ./ ref.fwhm_ref - 1));
ref.shoulder_lo = shoulder_lo;
ref.shoulder_hi = shoulder_hi;
ref.peak_wl_all = pk;
ref.centroid_all = ct;
ref.fwhm_all = fw;
end

function fwhm = fwhm_of(wl, prof, pk_row, pk_val)
base = min(prof);
half = base + 0.5*(pk_val - base);
l = find(prof(1:pk_row) <= half, 1, 'last');
r = find(prof(pk_row:end) <= half, 1, 'first');
if isempty(l), wl_l = wl(1); else
    if l == pk_row, wl_l = wl(l); else
        wl_l = interp1(prof([l l+1]), wl([l l+1]), half, 'linear');
    end
end
if isempty(r), wl_r = wl(end); else
    r = pk_row + r - 1;
    if r == pk_row, wl_r = wl(r); else
        wl_r = interp1(prof([r-1 r]), wl([r-1 r]), half, 'linear');
    end
end
fwhm = wl_r - wl_l;
if ~isfinite(fwhm) || fwhm <= 0
    fwhm = wl(end) - wl(1);
end
end
