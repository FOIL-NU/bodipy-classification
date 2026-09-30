function test_spectral_transforms()
% TEST_SPECTRAL_TRANSFORMS
% Mechanical validation of spectral_transforms.m on SYNTHETIC spectra with
% known peak positions, known widths and a known shoulder.
%
% This exists so the ablation procedure can be fixed and verified before it
% is ever run on real data. An ablation experiment's null result is only
% interpretable if the transform demonstrably removed the property it
% claims to remove: "accuracy survived peak alignment" means nothing if the
% alignment did not actually align the peaks. Each assertion below states
% the property the corresponding transform must establish.


% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
fprintf('Validating spectral transforms on synthetic spectra\n');
fprintf('===================================================\n');

% Real non-uniform axis, so the test exercises the same interpolation path
% as the real data rather than an idealized uniform grid.
d = load(fullfile(repo_root,'results','cross_validation','pairwise_reference_net.mat'));
wl = d.wl(:);
n_h = numel(wl);
n_w = 5;
n_s = 400;

rng(7, 'twister');
% The fixture deliberately mimics the REAL band shape: a steep blue rise and a
% long red tail (a split Gaussian), not a symmetric Gaussian. This matters for
% more than realism. A symmetric Gaussian broad enough to vary in width pushes
% its blue half-maximum below the 599.7 nm axis start, so FWHM becomes
% unmeasurable for a large fraction of molecules (measured: 207 of 400) and no
% width transform can equalize what cannot be measured. The real spectra have
% zero such failures, so a symmetric fixture would test the transforms under
% conditions strictly harsher than the data they are applied to.
true_peak = 613 + 10*rand(n_s,1);          % 613-623 nm, as measured
sig_blue = 8 + 4*rand(n_s,1);              % steep blue flank
sig_red = 10 + 15*rand(n_s,1);             % red tail; drives width variation.
% Calibrated so FWHM = 1.1774*(sig_blue+sig_red) spans ~21-45 nm, bracketing the
% measured 26-42 nm IQR, and so the 660 nm shoulder stays a bump on a decaying
% tail rather than overtaking the main peak (which happened with a broader tail
% and made the measured 'peak' jump to 660 nm for many molecules).
has_shoulder = rand(n_s,1) > 0.5;
rois = zeros(n_h, n_w, n_s);
for i = 1:n_s
    prof = zeros(n_h,1);
    blue = wl <= true_peak(i);
    prof(blue)  = 1000*exp(-0.5*((wl(blue)  - true_peak(i))/sig_blue(i)).^2);
    prof(~blue) = 1000*exp(-0.5*((wl(~blue) - true_peak(i))/sig_red(i)).^2);
    if has_shoulder(i)
        prof = prof + 250*exp(-0.5*((wl - 660)/8).^2);   % shoulder at 660 nm, ~0.5 of peak as measured
    end
    rois(:,:,i) = repmat(prof, 1, n_w) + 5*randn(n_h, n_w);
end

ref = reference_constants(rois, wl, 640, 680);
fprintf('Reference constants: wl_ref = %.1f nm, fwhm_ref = %.1f nm, shift_ref = %.1f nm, scale_ref = %.3f\n', ...
    ref.wl_ref, ref.fwhm_ref, ref.shift_ref, ref.scale_ref);

pass = true;

% ---- 1. peak_align must collapse the spread of peak positions ----
[out, inf1] = spectral_transforms(rois, wl, 'peak_align', ref);
pk_before = measure_peaks(rois, wl);
pk_after  = measure_peaks(out,  wl);
fprintf('\n[1] peak_align\n');
fprintf('    peak position std: %.2f nm before -> %.2f nm after (target ~0)\n', std(pk_before), std(pk_after));
fprintf('    mean |peak - wl_ref|: %.2f nm before -> %.2f nm after\n', ...
    mean(abs(pk_before - ref.wl_ref)), mean(abs(pk_after - ref.wl_ref)));
band_nm = mean(diff(wl));
pass = check(pass, std(pk_after) < band_nm && std(pk_after) < 0.5*std(pk_before), ...
    sprintf('peak_align aligns peaks to within one band (%.2f nm) and halves the spread', band_nm));
% and it must NOT destroy width
fw_before = measure_fwhm(rois, wl); fw_after = measure_fwhm(out, wl);
fprintf('    FWHM std (must be preserved): %.2f nm -> %.2f nm\n', std(fw_before), std(fw_after));
pass = check(pass, std(fw_after) > 0.6*std(fw_before), 'peak_align preserves width variation');

% ---- 2. width_norm must collapse the spread of FWHM ----
[out2, ~] = spectral_transforms(rois, wl, 'width_norm', ref);
fw2 = measure_fwhm(out2, wl);
pk2 = measure_peaks(out2, wl);
fprintf('\n[2] width_norm\n');
fprintf('    FWHM std: %.2f nm before -> %.2f nm after (target ~0)\n', std(fw_before), std(fw2));
fprintf('    peak position std (must be preserved): %.2f nm -> %.2f nm\n', std(pk_before), std(pk2));
pass = check(pass, std(fw2) < 0.35*std(fw_before), 'width_norm reduces FWHM spread by >2.8x');
pass = check(pass, std(pk2) > 0.7*std(pk_before), 'width_norm preserves peak-position variation');

% ---- 3. shoulder_remove must erase the shoulder contrast ----
[out3, ~] = spectral_transforms(rois, wl, 'shoulder_remove', ref);
sh_before = shoulder_excess(rois, wl);
sh_after  = shoulder_excess(out3, wl);
fprintf('\n[3] shoulder_remove\n');
fprintf('    shoulder excess, with-shoulder minus without: %.1f before -> %.1f after (target ~0)\n', ...
    mean(sh_before(has_shoulder)) - mean(sh_before(~has_shoulder)), ...
    mean(sh_after(has_shoulder)) - mean(sh_after(~has_shoulder)));
contrast_before = mean(sh_before(has_shoulder)) - mean(sh_before(~has_shoulder));
contrast_after  = mean(sh_after(has_shoulder)) - mean(sh_after(~has_shoulder));
pass = check(pass, abs(contrast_after) < 0.15*abs(contrast_before), 'shoulder_remove erases >85% of shoulder contrast');
% outside the window nothing should change
outside = wl < ref.shoulder_lo - 5 | wl > ref.shoulder_hi + 5;
delta_outside = max(abs(out3(outside,:,:) - rois(outside,:,:)), [], 'all');
fprintf('    max change outside the shoulder window: %.3g (target ~0)\n', delta_outside);
pass = check(pass, delta_outside < 1e-6, 'shoulder_remove leaves the rest of the band untouched');

% ---- 4. controls must preserve BETWEEN-molecule differences ----
[out4, ~] = spectral_transforms(rois, wl, 'const_shift', ref);
[out5, ~] = spectral_transforms(rois, wl, 'const_scale', ref);
pk4 = measure_peaks(out4, wl); fw5 = measure_fwhm(out5, wl);
fprintf('\n[4] resampling controls\n');
fprintf('    const_shift: peak-position std %.2f -> %.2f nm (must be preserved)\n', std(pk_before), std(pk4));
fprintf('    const_scale: FWHM std %.2f -> %.2f nm (must be preserved)\n', std(fw_before), std(fw5));
pass = check(pass, std(pk4) > 0.7*std(pk_before), 'const_shift preserves peak-position variation');
pass = check(pass, std(fw5) > 0.7*std(fw_before), 'const_scale preserves FWHM variation');

% ---- 4b. cumulative ablation must collapse BOTH properties ----
[out7, ~] = spectral_transforms(rois, wl, 'align_width', ref);
pk7 = measure_peaks(out7, wl); fw7 = measure_fwhm(out7, wl);
fprintf('\n[4b] align_width (cumulative)\n');
fprintf('    peak position std: %.2f -> %.2f nm (target ~0)\n', std(pk_before), std(pk7));
fprintf('    FWHM std:          %.2f -> %.2f nm (target ~0)\n', std(fw_before), std(fw7));
pass = check(pass, std(pk7) < band_nm && std(pk7) < 0.5*std(pk_before), ...
    'align_width aligns peaks to within one band and halves the spread');
pass = check(pass, std(fw7) < 0.35*std(fw_before), 'align_width collapses FWHM spread');

[out8, ~] = spectral_transforms(rois, wl, 'const_both', ref);
pk8 = measure_peaks(out8, wl); fw8 = measure_fwhm(out8, wl);
fprintf('    const_both control: peak std %.2f -> %.2f, FWHM std %.2f -> %.2f (both must be preserved)\n', ...
    std(pk_before), std(pk8), std(fw_before), std(fw8));
pass = check(pass, std(pk8) > 0.7*std(pk_before), 'const_both preserves peak-position variation');
pass = check(pass, std(fw8) > 0.7*std(fw_before), 'const_both preserves FWHM variation');

[out9, ~] = spectral_transforms(rois, wl, 'align_width_shoulder', ref);
sh9 = shoulder_excess(out9, wl);
c9 = mean(sh9(has_shoulder)) - mean(sh9(~has_shoulder));
fprintf('    align_width_shoulder: shoulder contrast %.1f -> %.1f\n', contrast_before, c9);
pass = check(pass, abs(c9) < 0.35*abs(contrast_before), 'align_width_shoulder also suppresses the shoulder');

% ---- 4c. centroid_align must equalize whole-band position ----
[outA, ~] = spectral_transforms(rois, wl, 'centroid_align', ref);
ct_before = measure_centroid(rois, wl);
ct_after  = measure_centroid(outA,  wl);
[outB, ~] = spectral_transforms(rois, wl, 'peak_align', ref);
ct_peakal = measure_centroid(outB, wl);
fprintf('\n[4c] centroid_align vs peak_align, on whole-band position\n');
fprintf('    centroid std: %.2f before -> %.2f after centroid_align (target ~0)\n', std(ct_before), std(ct_after));
fprintf('    centroid std after PEAK alignment: %.2f  (peak alignment need not equalize the centroid)\n', std(ct_peakal));
pass = check(pass, std(ct_after) < 0.35*std(ct_before), 'centroid_align collapses centroid spread');
pass = check(pass, std(ct_after) < std(ct_peakal), 'centroid_align equalizes centroid better than peak_align does');

% ---- 5. identity must be exact ----
[out6, ~] = spectral_transforms(rois, wl, 'identity', ref);
fprintf('\n[5] identity: max deviation %.3g\n', max(abs(out6 - rois), [], 'all'));
pass = check(pass, max(abs(out6 - rois), [], 'all') == 0, 'identity is exact');

pass = check(pass, inf1.n_fwhm_fallback < 0.10*n_s, 'FWHM measurable for >90% of fixture molecules (real data: 100%)');
fprintf('\nEdge clamping affected %.2f%% of output samples under peak_align; %d/%d molecules used an FWHM fallback.\n', ...
    100*inf1.frac_clamped, inf1.n_fwhm_fallback, n_s);

fprintf('\n%s\n', repmat('=', 1, 51));
if pass
    fprintf('ALL CHECKS PASSED -- transforms do what they claim.\n');
else
    error('VALIDATION FAILED -- do not run the ablation until this is fixed.');
end
end

function ok = check(prev, cond, label)
fprintf('    %s  %s\n', ternary(cond, 'PASS', 'FAIL'), label);
ok = prev && cond;
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end

function pk = measure_peaks(rois, wl)
n = size(rois,3); pk = nan(n,1);
for i = 1:n
    [~, r] = max(movmean(mean(rois(:,:,i),2),3));
    pk(i) = wl(r);
end
end

function fw = measure_fwhm(rois, wl)
n = size(rois,3); fw = nan(n,1);
for i = 1:n
    prof = movmean(mean(rois(:,:,i),2),3);
    [pv, pr] = max(prof);
    base = min(prof); half = base + 0.5*(pv-base);
    l = find(prof(1:pr) <= half, 1, 'last');
    r = find(prof(pr:end) <= half, 1, 'first');
    % Interpolated crossings, matching reference_constants.m. Nearest-band
    % crossings quantize to the ~3.72 nm grid, which inflates the apparent
    % residual spread after a width transform.
    if isempty(l)
        wl_l = wl(1);
    elseif l == pr
        wl_l = wl(l);
    else
        wl_l = interp1(prof([l l+1]), wl([l l+1]), half, 'linear');
    end
    if isempty(r)
        wl_r = wl(end);
    else
        r = pr + r - 1;
        if r == pr
            wl_r = wl(r);
        else
            wl_r = interp1(prof([r-1 r]), wl([r-1 r]), half, 'linear');
        end
    end
    if ~isfinite(wl_l), wl_l = wl(1); end
    if ~isfinite(wl_r), wl_r = wl(end); end
    fw(i) = wl_r - wl_l;
end
end

function ex = shoulder_excess(rois, wl)
% Intensity in the shoulder window above a straight line joining its
% endpoints -- the quantity 'shoulder_remove' is meant to zero out.
n = size(rois,3); ex = nan(n,1);
sel = find(wl >= 640 & wl <= 680);
for i = 1:n
    prof = mean(rois(:,:,i),2);
    base = interp1([wl(sel(1)) wl(sel(end))], [prof(sel(1)) prof(sel(end))], wl(sel), 'linear');
    ex(i) = sum(prof(sel) - base);
end
end

function ct = measure_centroid(rois, wl)
% Intensity-weighted mean wavelength, the quantity 'centroid_align' targets.
n = size(rois,3); ct = nan(n,1);
for i = 1:n
    w = max(movmean(mean(rois(:,:,i),2),3), 0);
    ct(i) = sum(wl .* w) / max(sum(w), eps);
end
end
