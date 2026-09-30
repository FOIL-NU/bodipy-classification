function occlusion_band_importance(varargin)
% OCCLUSION_BAND_IMPORTANCE
% Band-occlusion / band-permutation importance for the trained BODIPY
% CNN classifiers, measured in CLASSIFICATION terms (change in held-out
% accuracy and in the correct-class pre-softmax logit) rather than in
% activation terms.
%
% WHY THIS EXISTS, AND WHAT IT REPLACES
% -------------------------------------
% The reviewer asked which spectral features the network uses to
% DISCRIMINATE (peak position, shoulder intensity, or overall band
% shape). The earlier conv1-activation analysis (cnn_weight_analysis.m,
% reverify_interpretability*.m) cannot answer that, for a structural
% reason rather than a fixable bug: first-layer activation measures
% where the network RESPONDS, which is dominated by where the input is
% large. The round-2 random-initialization control demonstrated exactly
% this -- untrained He-initialized filters cluster in the same 599-625
% nm onset band as the trained ones -- and peak activation magnitude
% was found to be essentially uncorrelated with any filter's own class-
% separation score (r = 0.002, -0.030). No further auditing of that
% analysis can reach the reviewer's question, because discrimination is
% a property of the whole network's OUTPUT, not of conv1 response
% magnitude.
%
% Occluding a wavelength window and re-measuring accuracy is immune to
% that amplitude confound by construction: a high-variance region that
% carries no class information produces no accuracy drop when removed.
%
% RELATION TO THE EXISTING OCCLUSION FIGURE
% -----------------------------------------
% cnn_weight_analysis.m already runs MATLAB's occlusionSensitivity, but
% it (a) occludes one 3.7 nm row at a time rather than a spectrally
% meaningful window, (b) reports the change in a class SCORE, not in
% classification performance, (c) replaces each occluded row with that
% ROI's OWN mean intensity, which is a per-sample baseline that still
% leaks that sample's overall brightness, and (d) has no untrained
% control. This script differs on all four points and is intended to be
% the primary evidence for the interpretability comment; the existing
% figure is retained as a consistency check.
%
% BASELINE USED FOR OCCLUSION -- AND A LIMITATION
% -----------------------------------------------
% Occluded windows are replaced with the PER-BAND MEAN over the full
% brightness-filtered, outlier-removed population of the dyes involved
% (class-balanced), recomputed here from data/processed_combined
% with the same cutoff and rmoutliers filtering the CV scripts use.
%
% This is NOT literally the training-fold mean. The trained network's
% imageInputLayer stores its 'zerocenter' Mean as a SCALAR (verified:
% size [1 1]), so the per-band training mean is not recoverable from the
% saved net, and save_reference_net.m drew its fold with a RNG state
% that the saved artifacts do not pin down. The population mean is a
% split-independent stand-in for it, and is disclosed as such. Two
% further checks bound how much the choice matters:
%   * the whole sweep is repeated with the per-band mean of the held-out
%     set itself (a test-set statistic -- reported only as a sensitivity
%     check, never as the headline), and
%   * band-PERMUTATION importance, which needs no baseline at all: the
%     window's values are shuffled across samples, preserving that
%     band's marginal distribution exactly while destroying its relation
%     to the label.
% Agreement among the three is the evidence that the profile is not an
% artifact of the replacement value.
%
% CONTROLS
% --------
%   * Random-init control: the same sweep on untrained, He-initialized
%     networks of identical architecture. This controls for a real
%     confound -- that occluding certain windows might perturb ANY
%     network's output more than others (e.g. via the 'same'-padding
%     boundary rows, or simply because some bands carry more energy).
%     It does NOT establish statistical significance for the trained
%     network, because an untrained net sits near chance and its Delta-
%     accuracy has a different variance; significance is taken from the
%     paired bootstrap below instead.
%   * Paired bootstrap over held-out samples for a CI on each window's
%     Delta-accuracy, reported both raw and Bonferroni-adjusted across
%     the 55 windows.
%
% Usage:
%   occlusion_band_importance
%   occlusion_band_importance('window_nm', 20, 'n_random_trials', 30)

p = inputParser;
addParameter(p, 'window_nm', 15);          % primary window width
addParameter(p, 'alt_window_nm', 20);      % secondary width, reported for robustness
addParameter(p, 'n_random_trials', 20);
addParameter(p, 'n_perm_repeats', 20);
addParameter(p, 'n_boot', 2000);
% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addParameter(p, 'base_dir', repo_root);
parse(p, varargin{:});
opt = p.Results;

base_dir = opt.base_dir;
data_dir = fullfile(base_dir, 'data', 'processed_combined');
cutoff_val = 1000;                          % matches the CV scripts

nets = struct( ...
    'name',     {'pairwise', 'triplet'}, ...
    'ref_file', {fullfile(base_dir,'results','cross_validation','pairwise_reference_net.mat'), ...
                 fullfile(base_dir,'results','cross_validation','triplet_reference_net.mat')}, ...
    'dyes',     {[], []});

out_dir = fullfile(base_dir, 'results', 'interpretability', 'occlusion');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

rng(20260902, 'twister');

L = {};
L{end+1} = 'Band-Occlusion / Band-Permutation Importance (classification-based)';
L{end+1} = '====================================================================';
L{end+1} = sprintf('Primary window width: %g nm   Secondary: %g nm', opt.window_nm, opt.alt_window_nm);
L{end+1} = sprintf('Random-init control trials: %d   Permutation repeats: %d   Bootstrap resamples: %d', ...
    opt.n_random_trials, opt.n_perm_repeats, opt.n_boot);
L{end+1} = '';

all_res = struct();

for ni = 1:numel(nets)
    net_name = nets(ni).name;
    fprintf('\n================ %s network ================\n', net_name);
    ref = load(nets(ni).ref_file);
    net = ref.net;
    wl = ref.wl(:);
    roi_data = double(ref.roi_data);        % [H W 1 N]
    roi_labels = ref.roi_labels;
    n_h = numel(wl);
    n_w = size(roi_data, 2);
    n_samp = size(roi_data, 4);
    class_names = categories(roi_labels);
    n_classes = numel(class_names);
    y_true = double(roi_labels);            % 1..n_classes

    if isfield(ref, 'reference_pair')
        dyes = ref.reference_pair;
    elseif isfield(ref, 'reference_triad')
        dyes = ref.reference_triad;
    else
        dyes = [];
    end

    % Human-readable class labels for figure legends. The reference-net
    % savers build labels in dye order -- pairwise as categorical([zeros;
    % ones]) over cat(3, dye1_rois, dye2_rois), triplet as categorical
    % ([ones;2*ones;3*ones]) over class_rois in dyes order -- so class c
    % is dyes(c). Falls back to the raw category name if the reference
    % net did not record its dyes.
    class_disp = cell(n_classes,1);
    for c = 1:n_classes
        if numel(dyes) == n_classes
            class_disp{c} = sprintf('BODIPY-%d', dyes(c));
        else
            class_disp{c} = sprintf('class %s', class_names{c});
        end
    end

    L{end+1} = sprintf('=== Network: %s (dyes %s) ===', net_name, mat2str(dyes));
    L{end+1} = sprintf('Held-out samples: %d   Classes: %d   Spectral bands: %d (%.1f-%.1f nm, ~%.2f nm mean spacing)', ...
        n_samp, n_classes, n_h, wl(1), wl(end), mean(diff(wl)));

    % ---------------------------------------------------------------
    % Baselines for occlusion
    % ---------------------------------------------------------------
    [pop_band_mean, pop_note] = population_band_mean(data_dir, dyes, cutoff_val, n_h, n_w);
    heldout_band_mean = mean(reshape(permute(roi_data, [1 2 4 3]), n_h, []), 2); % [H x 1]
    L{end+1} = sprintf('Occlusion baseline (primary): %s', pop_note);
    if ~isempty(pop_band_mean)
        L{end+1} = sprintf('  Population vs held-out per-band mean: max abs difference %.3g (%.1f%% of the population mean''s own range)', ...
            max(abs(pop_band_mean - heldout_band_mean)), ...
            100*max(abs(pop_band_mean - heldout_band_mean))/range(pop_band_mean));
    end

    % ---------------------------------------------------------------
    % Baseline (unoccluded) performance
    % ---------------------------------------------------------------
    dlnet_logits = logits_network(net);
    [correct0, logit0] = eval_batch(net, dlnet_logits, roi_data, y_true);
    acc0 = mean(correct0);
    L{end+1} = sprintf('Unoccluded held-out accuracy: %.4f (%d/%d correct); mean correct-class logit %.4f', ...
        acc0, sum(correct0), n_samp, mean(logit0));

    % ---------------------------------------------------------------
    % Occlusion sweep -- primary window, population-mean baseline
    % ---------------------------------------------------------------
    win_rows = window_rows(wl, opt.window_nm);
    fprintf('Occlusion sweep (%g nm windows, population-mean baseline)...\n', opt.window_nm);
    [dacc, dlogit, correct_occ, dlogit_per_class] = occlusion_sweep( ...
        net, dlnet_logits, roi_data, y_true, win_rows, pop_band_mean, correct0, logit0, n_classes);

    % Sensitivity check: same sweep, held-out-set per-band mean
    fprintf('Occlusion sweep (held-out-mean baseline, sensitivity check)...\n');
    dacc_ho = occlusion_sweep(net, dlnet_logits, roi_data, y_true, win_rows, heldout_band_mean, correct0, logit0, n_classes);

    % Secondary window width
    win_rows_alt = window_rows(wl, opt.alt_window_nm);
    fprintf('Occlusion sweep (%g nm windows, robustness check)...\n', opt.alt_window_nm);
    dacc_alt = occlusion_sweep(net, dlnet_logits, roi_data, y_true, win_rows_alt, pop_band_mean, correct0, logit0, n_classes);

    % ---------------------------------------------------------------
    % Paired bootstrap CI on Delta-accuracy
    % ---------------------------------------------------------------
    fprintf('Paired bootstrap (%d resamples)...\n', opt.n_boot);
    boot = zeros(opt.n_boot, n_h);
    for b = 1:opt.n_boot
        idx = randi(n_samp, n_samp, 1);
        boot(b, :) = mean(correct0(idx)) - mean(correct_occ(idx, :), 1);
    end
    ci_lo   = prctile(boot,  2.5, 1)';
    ci_hi   = prctile(boot, 97.5, 1)';
    alpha_b = 0.05 / n_h;                                    % Bonferroni across windows
    ci_lo_b = prctile(boot, 100*alpha_b/2, 1)';
    ci_hi_b = prctile(boot, 100*(1-alpha_b/2), 1)';
    sig_raw = ci_lo > 0;
    sig_bon = ci_lo_b > 0;

    % ---------------------------------------------------------------
    % Band-permutation importance (baseline-free)
    % ---------------------------------------------------------------
    fprintf('Permutation importance (%d repeats)...\n', opt.n_perm_repeats);
    perm_dacc = zeros(n_h, opt.n_perm_repeats);
    for rep = 1:opt.n_perm_repeats
        for i = 1:n_h
            rows = win_rows{i};
            Xp = roi_data;
            sh = randperm(n_samp);
            Xp(rows, :, :, :) = roi_data(rows, :, :, sh);
            cp = eval_batch(net, dlnet_logits, Xp, y_true);
            perm_dacc(i, rep) = acc0 - mean(cp);
        end
    end
    perm_mean = mean(perm_dacc, 2);
    perm_std  = std(perm_dacc, 0, 2);

    % ---------------------------------------------------------------
    % Random-initialization control
    % ---------------------------------------------------------------
    fprintf('Random-init control (%d untrained networks)...\n', opt.n_random_trials);
    rand_dacc = zeros(n_h, opt.n_random_trials);
    rand_flip = zeros(n_h, opt.n_random_trials);
    rand_acc0 = zeros(opt.n_random_trials, 1);
    for t = 1:opt.n_random_trials
        dlr = randomize_weights(dlnet_logits);
        [c0r, ~, pred0r] = eval_batch_dl(dlr, roi_data, y_true);
        rand_acc0(t) = mean(c0r);
        for i = 1:n_h
            rows = win_rows{i};
            Xo = roi_data;
            Xo(rows, :, :, :) = repmat(pop_band_mean(rows), [1 n_w 1 n_samp]);
            [cir, ~, predir] = eval_batch_dl(dlr, Xo, y_true);
            rand_dacc(i, t) = rand_acc0(t) - mean(cir);
            rand_flip(i, t) = mean(predir ~= pred0r);
        end
    end
    rand_mean = mean(rand_dacc, 2);
    rand_sd   = std(rand_dacc, 0, 2);
    rand_flip_mean = mean(rand_flip, 2);

    % ---------------------------------------------------------------
    % Reference curves for feature attribution
    % ---------------------------------------------------------------
    class_mean_spec = zeros(n_h, n_classes);
    for c = 1:n_classes
        sel = y_true == c;
        class_mean_spec(:, c) = mean(reshape(permute(roi_data(:,:,:,sel), [1 2 4 3]), n_h, []), 2);
    end
    pooled_spec = mean(class_mean_spec, 2);
    amp_curve   = pooled_spec;                                    % amplitude confound reference
    if n_classes == 2
        diff_curve = abs(class_mean_spec(:,1) - class_mean_spec(:,2));
    else
        diff_curve = max(class_mean_spec, [], 2) - min(class_mean_spec, [], 2);
    end
    slope_curve = abs(gradient(pooled_spec, wl));                 % band-shape / edge reference

    r_amp   = corr(dacc, amp_curve,   'type', 'Spearman');
    r_diff  = corr(dacc, diff_curve,  'type', 'Spearman');
    r_slope = corr(dacc, slope_curve, 'type', 'Spearman');
    r_perm  = corr(dacc, perm_mean,   'type', 'Spearman');
    % Amplitude and between-class difference are themselves strongly
    % correlated across the spectral axis (both peak inside the emission
    % band), so a plain correlation cannot separate 'the network attends
    % where the signal is large' from 'the network attends where the
    % classes differ' -- which is precisely the confound that sank the
    % activation analysis. Rank partial correlations do separate them.
    Rd = tiedrank(dacc); Ra = tiedrank(amp_curve); Rc = tiedrank(diff_curve);
    r_amp_diff_xy = corr(amp_curve, diff_curve, 'type', 'Spearman');
    pr_diff = partialcorr(Rd, Rc, Ra);   % dacc vs class-difference, amplitude held constant
    pr_amp  = partialcorr(Rd, Ra, Rc);   % dacc vs amplitude, class-difference held constant
    r_hobase= corr(dacc, dacc_ho,     'type', 'Spearman');
    r_altw  = corr(dacc, dacc_alt,    'type', 'Spearman');

    % Peak / shoulder locations from the real class-mean spectra
    peak_wl = zeros(n_classes,1);
    for c = 1:n_classes
        [~, ix] = max(class_mean_spec(:,c));
        peak_wl(c) = wl(ix);
    end
    [sh_wl, sh_note] = shoulder_positions(roi_data, wl, y_true, n_classes);

    % Localization of the Delta-accuracy profile
    pos = max(dacc, 0);
    tot = sum(pos);
    [sorted_pos, order] = sort(pos, 'descend');
    frac_top3 = sum(sorted_pos(1:min(3,n_h))) / max(tot, eps);
    csum = cumsum(sorted_pos) / max(tot, eps);
    n50 = find(csum >= 0.5, 1);
    w50_nm = n50 * mean(diff(wl));       % total spectral width carrying 50% of the drop
    [dacc_max, imax] = max(dacc);
    argmax_wl = wl(imax);

    % ---------------------------------------------------------------
    % Report
    % ---------------------------------------------------------------
    L{end+1} = '';
    L{end+1} = sprintf('--- Occlusion sweep, %g nm windows, population-mean replacement ---', opt.window_nm);
    L{end+1} = sprintf('Largest accuracy drop: %.4f (%.1f accuracy points) at %.1f nm', dacc_max, 100*dacc_max, argmax_wl);
    L{end+1} = sprintf('  95%% bootstrap CI at that window: [%.4f, %.4f]%s', ci_lo(imax), ci_hi(imax), ...
        ternary_str(sig_bon(imax), '  (survives Bonferroni across all 55 windows)', '  (does NOT survive Bonferroni)'));
    L{end+1} = sprintf('Windows with a drop significant at raw 95%%: %d of %d; after Bonferroni: %d of %d', ...
        sum(sig_raw), n_h, sum(sig_bon), n_h);
    if any(sig_bon)
        L{end+1} = sprintf('  Bonferroni-significant windows (nm): %s', num2str(wl(sig_bon)', '%.1f '));
    end
    L{end+1} = sprintf('Random-init control: mean |Delta-accuracy| across windows %.4f (max %.4f); trained-network max is %.1fx that maximum.', ...
        mean(abs(rand_mean)), max(abs(rand_mean)), dacc_max / max(max(abs(rand_mean)), eps));
    L{end+1} = sprintf('  Untrained baseline accuracy across %d trials: %.3f +/- %.3f (chance = %.3f) -- the control bounds sweep artifacts, not significance.', ...
        opt.n_random_trials, mean(rand_acc0), std(rand_acc0), 1/n_classes);
    L{end+1} = sprintf('  Untrained prediction-flip rate: mean %.3f, max %.3f across windows (a more sensitive artifact probe than its Delta-accuracy).', ...
        mean(rand_flip_mean), max(rand_flip_mean));
    L{end+1} = '';
    L{end+1} = '--- Agreement across methods and choices (Spearman rho over the 55 windows) ---';
    L{end+1} = sprintf('  vs. permutation importance (no baseline value at all): rho = %+.3f', r_perm);
    L{end+1} = sprintf('  vs. same sweep with held-out-set mean as baseline:     rho = %+.3f', r_hobase);
    L{end+1} = sprintf('  vs. same sweep with %g nm windows:                     rho = %+.3f', opt.alt_window_nm, r_altw);
    L{end+1} = '';
    L{end+1} = '--- Which spectral feature does the drop track? (Spearman rho) ---';
    L{end+1} = sprintf('  vs. pooled mean spectral AMPLITUDE (the confound that invalidated the activation analysis): rho = %+.3f', r_amp);
    L{end+1} = sprintf('  vs. between-class mean-intensity DIFFERENCE:                                              rho = %+.3f', r_diff);
    L{end+1} = sprintf('  vs. |d(mean spectrum)/d(lambda)| (band-shape / edge steepness):                            rho = %+.3f', r_slope);
    L{end+1} = sprintf('  (Amplitude and between-class difference are themselves correlated across wavelength: rho = %+.3f,', r_amp_diff_xy);
    L{end+1} = '   so the plain correlations above cannot separate them. Rank PARTIAL correlations, which can:';
    L{end+1} = sprintf('     drop vs. between-class difference, amplitude held constant: %+.3f', pr_diff);
    L{end+1} = sprintf('     drop vs. amplitude, between-class difference held constant: %+.3f', pr_amp);
    L{end+1} = '   The random-init control is the other half of this argument: an untrained network sees the same';
    L{end+1} = '   amplitudes and shows no comparable drop, so amplitude alone does not produce the profile.';
    L{end+1} = sprintf('  Class peak positions (from class-mean spectra): %s nm', num2str(peak_wl', '%.1f '));
    L{end+1} = sprintf('  Shoulder positions: %s', sh_note);
    L{end+1} = sprintf('  Distance from largest-drop window to nearest class peak: %.1f nm', min(abs(argmax_wl - peak_wl)));
    if ~isempty(sh_wl) && any(~isnan(sh_wl))
        L{end+1} = sprintf('  Distance from largest-drop window to nearest detected shoulder: %.1f nm', ...
            min(abs(argmax_wl - sh_wl(~isnan(sh_wl)))));
    end
    L{end+1} = '';
    L{end+1} = '--- Localization: is discrimination in one feature or spread over the band? ---';
    L{end+1} = sprintf('  Fraction of the total accuracy drop in the top 3 windows: %.3f', frac_top3);
    L{end+1} = sprintf('  Number of windows carrying 50%% of the total drop: %d (approx %.0f nm of spectrum)', n50, w50_nm);
    L{end+1} = sprintf('  Interpretation threshold used: <=3 windows / <=%d nm for "localized to a single feature"; more than that reads as distributed band shape.', ...
        round(3*mean(diff(wl))));
    L{end+1} = '';

    % Per-class logit drop peaks
    L{end+1} = '--- Correct-class logit drop, per class ---';
    for c = 1:n_classes
        [mx, ix] = max(dlogit_per_class(:, c));
        L{end+1} = sprintf('  Class %s: largest mean logit drop %.4f at %.1f nm', class_names{c}, mx, wl(ix));
    end
    L{end+1} = '';

    % ---------------------------------------------------------------
    % Figures
    % ---------------------------------------------------------------
    cc = lines(max(n_classes,3));

    figure('Position', [100 100 1000 560]);
    yyaxis left; hold on;
    h_ci = fill([wl; flipud(wl)], [ci_lo; flipud(ci_hi)], [0.2 0.4 0.8], ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none');
    h_rand = fill([wl; flipud(wl)], [rand_mean-2*rand_sd; flipud(rand_mean+2*rand_sd)], [0.5 0.5 0.5], ...
        'FaceAlpha', 0.25, 'EdgeColor', 'none');
    h_tr = plot(wl, dacc, '-', 'LineWidth', 2.2, 'Color', [0.1 0.25 0.7]);
    plot(wl, rand_mean, '-', 'LineWidth', 1.2, 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
    yline(0, 'k--', 'HandleVisibility', 'off');
    ax = gca; ax.YAxis(1).Color = 'k';
    xlabel('Wavelength (nm)');
    ylabel(sprintf('Drop in held-out accuracy when %g nm window is occluded', opt.window_nm));
    h_spec = overlay_spectra(wl, class_mean_spec, n_classes, cc, class_disp);
    % Legend built from explicit handles: the spectra live on the right
    % y-axis, and an auto legend on a yyaxis plot does not reliably pick
    % up both sides.
    legend([h_tr, h_ci, h_rand, h_spec], ...
        [{'trained network', '95% bootstrap CI', 'random-init control (mean +/- 2 SD)'}, ...
         cellfun(@(nm) sprintf('%s mean spectrum (right axis)', nm), class_disp(:)', 'UniformOutput', false)], ...
        'Location', 'best');
    title(sprintf('%s network: band-occlusion accuracy drop vs. wavelength', net_name));
    grid on;
    save_figure(gcf, fullfile(out_dir, sprintf('occlusion_accuracy_drop_%s', net_name)));

    figure('Position', [100 100 1000 560]);
    yyaxis left; hold on;
    h_cls = gobjects(n_classes,1);
    for c = 1:n_classes
        h_cls(c) = plot(wl, dlogit_per_class(:, c), '-', 'LineWidth', 2, 'Color', cc(c,:));
    end
    yline(0, 'k--', 'HandleVisibility', 'off');
    ax = gca; ax.YAxis(1).Color = 'k';
    xlabel('Wavelength (nm)'); ylabel('Drop in correct-class pre-softmax logit');
    h_spec = overlay_spectra(wl, class_mean_spec, n_classes, cc, class_disp);
    legend([h_cls(:)', h_spec], ...
        [cellfun(@(nm) sprintf('%s logit drop', nm), class_disp(:)', 'UniformOutput', false), ...
         cellfun(@(nm) sprintf('%s mean spectrum (right axis)', nm), class_disp(:)', 'UniformOutput', false)], ...
        'Location', 'best');
    title(sprintf('%s network: correct-class logit drop vs. wavelength', net_name));
    grid on;
    save_figure(gcf, fullfile(out_dir, sprintf('occlusion_logit_drop_%s', net_name)));

    figure('Position', [100 100 1000 560]); hold on;
    plot(wl, dacc, '-', 'LineWidth', 2, 'DisplayName', sprintf('occlusion, population mean, %g nm', opt.window_nm));
    plot(wl, dacc_ho, '--', 'LineWidth', 1.5, 'DisplayName', 'occlusion, held-out mean baseline');
    plot(wl, dacc_alt, ':', 'LineWidth', 1.8, 'DisplayName', sprintf('occlusion, %g nm windows', opt.alt_window_nm));
    errorbar(wl, perm_mean, perm_std, '-', 'LineWidth', 1.2, 'DisplayName', 'permutation importance (no baseline)');
    yline(0, 'k--', 'HandleVisibility', 'off');
    xlabel('Wavelength (nm)'); ylabel('Drop in held-out accuracy');
    legend('Location', 'best'); grid on;
    title(sprintf('%s network: robustness of the importance profile to method and baseline', net_name));
    save_figure(gcf, fullfile(out_dir, sprintf('occlusion_robustness_%s', net_name)));

    % ---------------------------------------------------------------
    % Per-window CSV
    % ---------------------------------------------------------------
    T = table(wl, dacc, ci_lo, ci_hi, ci_lo_b, ci_hi_b, sig_raw, sig_bon, dlogit, ...
        perm_mean, perm_std, rand_mean, rand_sd, dacc_ho, dacc_alt, ...
        amp_curve, diff_curve, slope_curve, ...
        'VariableNames', {'wavelength_nm','delta_accuracy','ci95_lo','ci95_hi', ...
            'ci_bonf_lo','ci_bonf_hi','sig_raw95','sig_bonferroni','delta_correct_class_logit', ...
            'perm_importance_mean','perm_importance_std','randinit_delta_acc_mean','randinit_delta_acc_sd', ...
            'delta_accuracy_heldout_baseline','delta_accuracy_altwindow', ...
            'pooled_mean_spectrum','between_class_diff','abs_spectral_slope'});
    writetable(T, fullfile(out_dir, sprintf('occlusion_band_importance_%s.csv', net_name)));

    r = struct('wl', wl, 'dacc', dacc, 'ci_lo', ci_lo, 'ci_hi', ci_hi, ...
        'sig_raw', sig_raw, 'sig_bon', sig_bon, 'dlogit', dlogit, ...
        'dlogit_per_class', dlogit_per_class, 'perm_mean', perm_mean, 'perm_std', perm_std, ...
        'rand_mean', rand_mean, 'rand_sd', rand_sd, 'rand_flip_mean', rand_flip_mean, ...
        'dacc_ho', dacc_ho, 'dacc_alt', dacc_alt, 'acc0', acc0, ...
        'class_mean_spec', class_mean_spec, 'class_names', {class_names}, ...
        'r_amp', r_amp, 'r_diff', r_diff, 'r_slope', r_slope, 'r_perm', r_perm, ...
        'partial_r_diff_given_amp', pr_diff, 'partial_r_amp_given_diff', pr_amp, ...
        'frac_top3', frac_top3, 'n50', n50, 'w50_nm', w50_nm, 'argmax_wl', argmax_wl, ...
        'peak_wl', peak_wl, 'shoulder_wl', sh_wl, 'dyes', dyes);
    all_res.(net_name) = r; %#ok<STRNU>
end

save(fullfile(out_dir, 'occlusion_band_importance_results.mat'), '-struct', 'all_res');

fid = fopen(fullfile(out_dir, 'occlusion_band_importance_summary.txt'), 'w');
fprintf(fid, '%s\n', L{:});
fclose(fid);
fprintf('%s\n', L{:});
fprintf('\nOutputs written to %s\n', out_dir);

end

% =====================================================================
% Helpers
% =====================================================================

function rows = window_rows(wl, width_nm)
% Windows are defined in NANOMETRES, not in rows, because the spectral
% axis is not uniformly sampled (mean ~3.72 nm, std ~1.19 nm), so a
% fixed row count would be a varying spectral width.
n = numel(wl);
rows = cell(n,1);
for i = 1:n
    rows{i} = find(abs(wl - wl(i)) <= width_nm/2);
end
end

function [dacc, dlogit, correct_occ, dlogit_per_class] = occlusion_sweep( ...
        net, dlnet_logits, X, y_true, win_rows, band_mean, correct0, logit0, n_classes)
n_h = numel(win_rows);
n_w = size(X, 2);
n_s = size(X, 4);
acc0 = mean(correct0);
dacc = zeros(n_h, 1);
dlogit = zeros(n_h, 1);
correct_occ = false(n_s, n_h);
dlogit_per_class = zeros(n_h, n_classes);
for i = 1:n_h
    rows = win_rows{i};
    Xo = X;
    Xo(rows, :, :, :) = repmat(band_mean(rows), [1 n_w 1 n_s]);
    [ci, li] = eval_batch(net, dlnet_logits, Xo, y_true);
    correct_occ(:, i) = ci;
    dacc(i) = acc0 - mean(ci);
    dlogit(i) = mean(logit0 - li);
    for c = 1:n_classes
        sel = y_true == c;
        dlogit_per_class(i, c) = mean(logit0(sel) - li(sel));
    end
end
end

function [correct, logit_true, pred] = eval_batch(net, dlnet_logits, X, y_true)
% Predicted label from the trained DAGNetwork's softmax head, and the
% pre-softmax logit of the TRUE class from the logits-only graph.
probs = predict(net, X);                 % [N x n_classes]
[~, pred] = max(probs, [], 2);
correct = pred == y_true(:);
Z = extractdata(predict(dlnet_logits, dlarray(single(X), 'SSCB')));  % [n_classes x N]
idx = sub2ind(size(Z), y_true(:)', 1:numel(y_true));
logit_true = Z(idx)';
end

function [correct, logit_true, pred] = eval_batch_dl(dlnet_logits, X, y_true)
% Same, but for an untrained dlnetwork with no softmax head available:
% argmax of the logits is identical to argmax of the softmax.
Z = extractdata(predict(dlnet_logits, dlarray(single(X), 'SSCB')));  % [n_classes x N]
[~, pred] = max(Z, [], 1);
pred = pred(:);
correct = pred == y_true(:);
idx = sub2ind(size(Z), y_true(:)', 1:numel(y_true));
logit_true = Z(idx)';
end

function dlnet = logits_network(net)
layers = net.Layers;
cls_idx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.ClassificationOutputLayer'), layers), 1);
sm_idx  = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.SoftmaxLayer'), layers), 1);
lg = layerGraph(net);
lg = removeLayers(lg, {layers(cls_idx).Name, layers(sm_idx).Name});
dlnet = dlnetwork(lg);
end

function dlnet = randomize_weights(dlnet)
% He-initialize every convolution/fully-connected weight, zero the
% biases, and reset batch-norm to identity (offset 0, scale 1, trained
% statistics 0/1). Matches the He initialization used by the round-2
% random-init control in reverify_interpretability_round2.m, so the two
% controls are on the same footing.
Lr = dlnet.Learnables;
for k = 1:height(Lr)
    pname = string(Lr.Parameter(k));
    v = Lr.Value{k};
    sz = size(v);
    switch pname
        case "Weights"
            fan_in = max(prod(sz(1:end-1)), 1);
            Lr.Value{k} = dlarray(single(randn(sz)) * single(sqrt(2/fan_in)));
        case "Bias"
            Lr.Value{k} = dlarray(zeros(sz, 'single'));
        case "Offset"
            Lr.Value{k} = dlarray(zeros(sz, 'single'));
        case "Scale"
            Lr.Value{k} = dlarray(ones(sz, 'single'));
    end
end
dlnet.Learnables = Lr;

St = dlnet.State;
for k = 1:height(St)
    pname = string(St.Parameter(k));
    sz = size(St.Value{k});
    if pname == "TrainedMean"
        St.Value{k} = dlarray(zeros(sz, 'single'));
    elseif pname == "TrainedVariance"
        St.Value{k} = dlarray(ones(sz, 'single'));
    end
end
dlnet.State = St;
end

function [band_mean, note] = population_band_mean(data_dir, dyes, cutoff_val, n_h, n_w) %#ok<INUSD>
% Per-band mean over the full brightness-filtered, outlier-removed
% population of the dyes involved, class-balanced (each dye contributes
% equally regardless of how many molecules survived filtering).
band_mean = [];
note = '';
if isempty(dyes)
    note = 'FALLBACK: reference net did not record which dyes it was trained on; using held-out-set per-band mean.';
    return;
end
per_dye = nan(n_h, numel(dyes));
for k = 1:numel(dyes)
    f = fullfile(data_dir, sprintf('bodipy%d.mat', dyes(k)));
    if ~exist(f, 'file')
        note = sprintf('FALLBACK: %s not found; using held-out-set per-band mean.', f);
        band_mean = [];
        return;
    end
    d = load(f);
    sel = d.n_photons_combined > cutoff_val;
    rois = d.spectral_rois_combined(:,:,sel);
    centroid = d.spectral_centroids_combined(sel);
    [~, rm] = rmoutliers(centroid, 'mean', 'thresholdfactor', 4);
    rois = double(rois(:,:,~rm));
    per_dye(:, k) = mean(reshape(rois, size(rois,1), []), 2);
end
band_mean = mean(per_dye, 2);
note = sprintf(['per-band mean over the full filtered population of dyes %s ' ...
    '(n_photons > %d, centroid rmoutliers 4 SD), class-balanced across dyes. ' ...
    'Split-independent stand-in for the training-fold mean, which the saved net does not record ' ...
    '(its imageInputLayer stores a SCALAR zerocenter mean).'], mat2str(dyes), cutoff_val);
end

function [sh_wl, note] = shoulder_positions(roi_data, wl, y_true, n_classes)
% Per-class median shoulder position, using the same detector as
% Approach 1 (extract_spectral_features.m), so the two analyses name the
% shoulder the same way.
sh_wl = nan(n_classes, 1);
frac = nan(n_classes, 1);
try
    profiles = squeeze(mean(roi_data, 2));      % [H x N]
    [F, names] = extract_spectral_features(profiles, wl);
    ix = find(contains(lower(names), 'shoulder') & contains(lower(names), 'position'), 1);
    if isempty(ix)
        note = 'shoulder feature not found in extract_spectral_features output';
        return;
    end
    for c = 1:n_classes
        v = F(y_true == c, ix);
        v = v(isfinite(v) & v > 0);
        frac(c) = numel(v) / sum(y_true == c);
        if ~isempty(v)
            sh_wl(c) = median(v);
        end
    end
    parts = strings(n_classes,1);
    for c = 1:n_classes
        if isnan(sh_wl(c))
            parts(c) = sprintf('class %d: none detected', c);
        else
            parts(c) = sprintf('class %d: %.1f nm (detected in %.0f%% of molecules)', c, sh_wl(c), 100*frac(c));
        end
    end
    note = strjoin(cellstr(parts), '; ');
catch ME
    note = sprintf('shoulder detection failed (%s)', ME.message);
end
end

function h = overlay_spectra(wl, class_mean_spec, n_classes, cc, class_disp)
% Overlays each class's mean real spectrum (dotted) on the right-hand
% y-axis, in the SAME colour as that class's left-axis curve, and
% returns the line handles so the caller can name them in the legend.
% Colours must be set explicitly on both sides: under yyaxis, MATLAB
% does not auto-cycle a distinct colour per line the way plain axes do.
yyaxis right; hold on;
% Normalized to a SHARED maximum (the largest value over all classes and
% bands), not per-class: dividing each dye by its own peak would force
% all curves to 1 and erase the genuine amplitude differences between
% dyes, which is information the reader needs -- amplitude vs. class
% difference is exactly what the partial correlations in this analysis
% separate. A shared divisor puts the axis on 0-1 while preserving the
% relative heights.
scale = max(class_mean_spec(:));
if ~isfinite(scale) || scale <= 0
    scale = 1;
end
h = gobjects(1, n_classes);
for c = 1:n_classes
    h(c) = plot(wl, class_mean_spec(:,c) / scale, ':', 'LineWidth', 1.5, 'Color', cc(c,:));
end
ylim([0 1]);
% UNITS: the CNN input is spectral_rois_combined, which step1_crop_data_batch.m
% builds as background-subtracted raw camera frames (`double(data) -
% polyval(bg_fit, bg_data)`, negatives clipped) where `data` was set to
% uint32(raw)*2. No gain or quantum-efficiency conversion is applied, so the
% underlying values are UNCALIBRATED detector counts, NOT photons -- and they
% carry the factor of 2 that spectral_profiles divides back out, so they are
% not on the same scale as spectral_profiles either. The photon counts used
% for the brightness cutoff (n_photons_combined) are a separate derived
% estimate. Because the raw scale is therefore uninterpretable, and this axis
% serves only to show where the emission band sits relative to the importance
% curve, the curves are plotted normalized rather than in counts.
ylabel('Mean spectral intensity (normalized to shared maximum)');
ax = gca; ax.YAxis(2).Color = 'k';
yyaxis left;
end

function s = ternary_str(cond, a, b)
if cond, s = a; else, s = b; end
end
