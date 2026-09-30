function input_ablation_experiments(varargin)
% INPUT_ABLATION_EXPERIMENTS
% Tests which spectral properties the CNN needs, by destroying one property
% at a time in the input and retraining from scratch.
%
% WHAT A NONZERO CONTRAST DOES AND DOES NOT ESTABLISH
% Each condition applies a transformation that TARGETS one spectral
% property, but no transformation is perfectly selective. Shoulder removal,
% for instance, measurably alters FWHM (12.65 -> 16.70 nm in the pairwise
% data) because flattening the shoulder changes the profile minimum against
% which the half-maximum is measured; the cumulative transform increases
% centroid spread rather than reducing it. A nonzero contrast therefore
% establishes that THE FEATURE-TARGETING TRANSFORMATION CAUSES A MEASURABLE
% ACCURACY LOSS, not that the targeted property is uniquely responsible.
% The wording of every verdict below reflects this, and the residual spread
% of peak, FWHM and centroid is reported for every condition so the reader
% can see which properties each transform actually moved.
%
% PROPERTIES TARGETED, NAMED FOR THE QUANTITY ACTUALLY EQUALIZED
%   peak_align      the PEAK-MAXIMUM wavelength. Does NOT equalize
%                   whole-band position: the centroid-minus-peak offset
%                   varies between molecules, so centroid spread only falls
%                   from 5.19 to 4.20 nm (pairwise), against 0.95 nm under
%                   centroid_align.
%   centroid_align  whole-band position (intensity-weighted mean
%                   wavelength). Separate ablation because the centroid is
%                   the strongest single descriptor in the data.
%   width_norm      FWHM, a WIDTH PROXY -- not the reviewer's broader
%                   "overall band shape" category, which FWHM does not
%                   exhaust.
%   shoulder_remove the 640-680 nm shoulder window.
%   align_width, align_width_shoulder  cumulative combinations.
%
% INFERENCE DESIGN
% The design is repeated across n_seeds independent runs, each with a fresh
% initialization, molecule subsample and partition. Inference is on the
% ACROSS-SEED paired difference (control minus ablation), never on the
% five-fold spread within a single run: folds are not independently trained
% conditions and their spread is not a valid error term. An earlier version
% used it as one and reached the opposite conclusion for every
% single-property contrast.
%
% MULTIPLICITY. Two pre-specified families are tested: the eight
% single-property contrasts (four properties x two dye sets) and the four
% cumulative contrasts. Holm-Bonferroni correction is applied WITHIN each
% family. Both nominal and Holm-adjusted p-values are reported, so a
% marginal nominal effect cannot be mistaken for a corrected one.
%
% CROSS-VALIDATION OF THE TRANSFORM ITSELF
% The transforms need dataset-wide reference constants (a common peak
% wavelength, FWHM and centroid). Computing these from all molecules would
% use held-out covariates to define the transformation -- not label
% leakage, but not strict cross-validation either. With ref_mode
% 'trainfold' (the default) each fold's constants are computed from that
% fold's TRAINING molecules only and then applied to both its training and
% test data. ref_mode 'pooled' reproduces the earlier, laxer behaviour and
% exists so the two can be compared as a sensitivity analysis.

p = inputParser;
% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addParameter(p, 'base_dir', repo_root);
addParameter(p, 'k_folds', 5);
addParameter(p, 'n_seeds', 5);
addParameter(p, 'max_epochs', 200);
addParameter(p, 'shoulder_lo', 640);
addParameter(p, 'shoulder_hi', 680);
addParameter(p, 'ref_mode', 'trainfold');   % 'trainfold' (strict) | 'pooled'
addParameter(p, 'tag', '');                 % suffix for output filenames
addParameter(p, 'replot_only', false);      % redraw figures from saved results, no retraining
parse(p, varargin{:});
opt = p.Results;

base_dir = opt.base_dir;
data_dir = fullfile(base_dir, 'data', 'processed_combined');
cutoff_val = 1000;
out_dir = fullfile(base_dir, 'results', 'interpretability', 'ablation');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
tag = opt.tag;

try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

cases = struct('name', {'pairwise','triplet'}, ...
               'dyes', {[4 10], [2 11 13]}, ...
               'n_per_dye', {600, 610});

conditions = {'identity','const_shift','peak_align','centroid_align', ...
              'const_scale','width_norm','shoulder_remove', ...
              'const_both','align_width','align_width_shoulder'};

cond_desc = containers.Map(conditions, { ...
    'unmodified spectra (baseline)', ...
    'RESAMPLING CONTROL for the alignment transforms (same shift for every molecule)', ...
    'targets the peak-maximum wavelength', ...
    'targets whole-band position (intensity-weighted mean wavelength)', ...
    'RESAMPLING CONTROL for width_norm (same width factor for every molecule)', ...
    'targets FWHM, a width proxy, about each molecule''s own peak', ...
    sprintf('targets the shoulder (%g-%g nm replaced by a straight line)', opt.shoulder_lo, opt.shoulder_hi), ...
    'RESAMPLING CONTROL for the cumulative transforms (same shift and width factor for every molecule)', ...
    'CUMULATIVE: peak maximum AND FWHM both targeted', ...
    'CUMULATIVE: peak maximum, FWHM AND shoulder all targeted'});

single_pairs = { ...
    {'peak_align',      'const_shift', 'peak-maximum wavelength'}, ...
    {'centroid_align',  'const_shift', 'whole-band position (centroid)'}, ...
    {'width_norm',      'const_scale', 'FWHM (width proxy)'}, ...
    {'shoulder_remove', 'identity',    'shoulder'}};
cumul_pairs = { ...
    {'align_width',          'const_both', 'peak maximum + FWHM jointly'}, ...
    {'align_width_shoulder', 'const_both', 'peak maximum + FWHM + shoulder jointly'}};

% Redraw from saved results. Deliberately does NOT rewrite the summary or
% results file: an earlier version did, and overwrote a completed 2.7-hour
% run's outputs with the empty structures of a replot pass.
if opt.replot_only
    R = load(fullfile(out_dir, sprintf('input_ablation_results%s.mat', tag)));
    for ci = 1:numel(cases)
        cname = cases(ci).name;
        if ~isfield(R, cname), continue; end
        make_ablation_figure(R.(cname).conditions, R.(cname), cname, R.(cname).dyes, out_dir, tag);
        fprintf('  %s: figures redrawn from saved results\n', cname);
    end
    return;
end

store = struct();

for ci = 1:numel(cases)
    cname = cases(ci).name;
    dyes = cases(ci).dyes;
    n_per_dye = cases(ci).n_per_dye;
    n_classes = numel(dyes);
    fprintf('\n================ %s: BODIPY %s ================\n', cname, mat2str(dyes));

    pop = cell(n_classes,1);
    wl = [];
    for k = 1:n_classes
        d = load(fullfile(data_dir, sprintf('bodipy%d.mat', dyes(k))));
        if isempty(wl), wl = d.wl(:); end
        sel = d.n_photons_combined > cutoff_val;
        r = d.spectral_rois_combined(:,:,sel);
        cen = d.spectral_centroids_combined(sel);
        [~, rm] = rmoutliers(cen, 'mean', 'thresholdfactor', 4);
        pop{k} = double(r(:,:,~rm));
    end

    acc_mean = nan(numel(conditions), opt.n_seeds);
    acc_folds = nan(numel(conditions), opt.k_folds, opt.n_seeds);
    diag_rows = {};

    for s = 1:opt.n_seeds
        rng(4200 + s, 'twister');
        rois = []; labels = [];
        for k = 1:n_classes
            idx = randperm(size(pop{k},3), n_per_dye);
            rois = cat(3, rois, pop{k}(:,:,idx));
            labels = [labels; k*ones(n_per_dye,1)]; %#ok<AGROW>
        end
        y = categorical(labels);
        cv = cvpartition(y, 'KFold', opt.k_folds, 'Stratify', true);

        for fold = 1:opt.k_folds
            tr = cv.training(fold); te = cv.test(fold);
            % Reference constants from TRAINING molecules only, then applied
            % to both training and test data of this fold.
            switch opt.ref_mode
                case 'trainfold'
                    ref_f = reference_constants(rois(:,:,tr), wl, opt.shoulder_lo, opt.shoulder_hi);
                case 'pooled'
                    ref_f = reference_constants(rois, wl, opt.shoulder_lo, opt.shoulder_hi);
                otherwise
                    error('Unknown ref_mode %s', opt.ref_mode);
            end
            for q = 1:numel(conditions)
                cond = conditions{q};
                [O, I] = spectral_transforms(rois, wl, cond, ref_f);
                X = reshape(O, [size(O,1), size(O,2), 1, size(O,3)]);
                if s == 1 && fold == 1
                    [psd, fsd, csd] = property_spread(O, wl);
                    diag_rows{end+1} = sprintf('  %-21s %-13.2f %-13.2f %-15.2f %s', ...
                        cond, psd, fsd, csd, sprintf('%.1f%%', 100*I.frac_clamped)); %#ok<AGROW>
                end
                net = train_one(X(:,:,:,tr), y(tr), n_classes, opt.max_epochs);
                acc_folds(q, fold, s) = mean(classify(net, X(:,:,:,te)) == y(te));
            end
            fprintf('  seed %d fold %d done\n', s, fold);
        end
        acc_mean(:, s) = mean(acc_folds(:, :, s), 2);
        for q = 1:numel(conditions)
            fprintf('    seed %d  %-21s %.3f\n', s, conditions{q}, acc_mean(q,s));
        end
    end

    store.(cname) = struct('dyes', dyes, 'n_classes', n_classes, 'chance', 1/n_classes, ...
        'conditions', {conditions}, 'acc_mean_per_seed', acc_mean, 'acc_folds', acc_folds, ...
        'diag_rows', {diag_rows}, 'n_seeds', opt.n_seeds, 'k_folds', opt.k_folds, ...
        'n_molecules', n_classes*n_per_dye);
end

% ---------------------------------------------------------------------
% Inference, after all cases are collected, so Holm can be applied across
% the whole pre-specified family rather than within one dye set.
% ---------------------------------------------------------------------
[sing, cum] = deal(struct('label', {}, 'case', {}, 'ablation', {}, 'control', {}, ...
    'property', {}, 'd', {}, 'mean', {}, 'ci_lo', {}, 'ci_hi', {}, 'p', {}));

for ci = 1:numel(cases)
    cname = cases(ci).name;
    r = store.(cname);
    for q = 1:numel(single_pairs)
        sing(end+1) = contrast(r, conditions, single_pairs{q}, cname, opt.n_seeds); %#ok<AGROW>
    end
    for q = 1:numel(cumul_pairs)
        cum(end+1) = contrast(r, conditions, cumul_pairs{q}, cname, opt.n_seeds); %#ok<AGROW>
    end
end
p_sing = holm([sing.p]);
p_cum  = holm([cum.p]);

L = {};
L{end+1} = 'Input-Ablation Experiments: which spectral properties does the CNN need?';
L{end+1} = '========================================================================';
L{end+1} = sprintf('%d independent runs x stratified %d-fold CV x %d epochs; architecture and hyperparameters', ...
    opt.n_seeds, opt.k_folds, opt.max_epochs);
L{end+1} = 'identical to the CNN validation. Only the input transform differs between conditions.';
L{end+1} = sprintf('Reference-constant mode: %s.', opt.ref_mode);
if strcmp(opt.ref_mode,'trainfold')
    L{end+1} = '  Each fold''s common reference constants are computed from that fold''s TRAINING molecules only and';
    L{end+1} = '  applied to both its training and test data, so no held-out covariate defines the transformation.';
end
L{end+1} = '';
L{end+1} = 'INFERENCE: across-seed paired difference (control minus ablation), two-sided 95% t CI, n = runs.';
L{end+1} = 'Fold spread within a run is NOT the error term. Holm-Bonferroni correction is applied within each of';
L{end+1} = 'two pre-specified families: the 8 single-property contrasts, and the 4 cumulative contrasts.';
L{end+1} = '';
L{end+1} = 'WHAT A NONZERO CONTRAST MEANS: the feature-targeting TRANSFORMATION causes a measurable accuracy loss.';
L{end+1} = 'No transform is perfectly selective (shoulder removal alters FWHM; the cumulative transform increases';
L{end+1} = 'centroid spread), so a contrast does not isolate the targeted property as uniquely responsible.';
L{end+1} = '';

for ci = 1:numel(cases)
    cname = cases(ci).name;
    r = store.(cname);
    L{end+1} = sprintf('=== %s: BODIPY %s, %d molecules per run, chance = %.3f ===', ...
        cname, strjoin(compose('%d', r.dyes), '/'), r.n_molecules, r.chance);
    L{end+1} = '';
    L{end+1} = 'Transform diagnostics (seed 1 fold 1; residual spread of each property AFTER the transform):';
    L{end+1} = sprintf('  %-21s %-13s %-13s %-15s %s', 'condition', 'peak sd (nm)', 'FWHM sd (nm)', 'centroid sd (nm)', 'clamped');
    L = [L, r.diag_rows]; %#ok<AGROW>
    L{end+1} = '  Note: peak_align leaves centroid spread largely intact, and shoulder_remove changes FWHM. Read each';
    L{end+1} = '  contrast against the row showing what its transform actually moved.';
    L{end+1} = '';
    L{end+1} = sprintf('Accuracy (mean over %d folds, then mean +/- sd over %d runs):', r.k_folds, r.n_seeds);
    for q = 1:numel(conditions)
        L{end+1} = sprintf('  %-21s %.3f +/- %.3f        %s', conditions{q}, ...
            mean(r.acc_mean_per_seed(q,:)), std(r.acc_mean_per_seed(q,:)), cond_desc(conditions{q}));
    end
    L{end+1} = '';
end

L{end+1} = 'SINGLE-PROPERTY CONTRASTS (family of 8, Holm-corrected):';
L{end+1} = sprintf('  %-10s %-32s %-9s %-22s %-9s %-9s %s', 'case', 'targeted property', 'diff', '95% CI', 'p', 'p_Holm', 'verdict at Holm 0.05');
for i = 1:numel(sing)
    L{end+1} = sprintf('  %-10s %-32s %+.3f    [%+.3f, %+.3f]    %.4f    %.4f    %s', ...
        sing(i).case, sing(i).property, sing(i).mean, sing(i).ci_lo, sing(i).ci_hi, ...
        sing(i).p, p_sing(i), verdict_str(p_sing(i) < 0.05));
end
L{end+1} = '';
L{end+1} = 'CUMULATIVE CONTRASTS (family of 4, Holm-corrected):';
L{end+1} = sprintf('  %-10s %-32s %-9s %-22s %-9s %-9s %s', 'case', 'targeted properties', 'diff', '95% CI', 'p', 'p_Holm', 'verdict at Holm 0.05');
for i = 1:numel(cum)
    L{end+1} = sprintf('  %-10s %-32s %+.3f    [%+.3f, %+.3f]    %.4f    %.4f    %s', ...
        cum(i).case, cum(i).property, cum(i).mean, cum(i).ci_lo, cum(i).ci_hi, ...
        cum(i).p, p_cum(i), verdict_str(p_cum(i) < 0.05));
end
L{end+1} = '';

for ci = 1:numel(cases)
    cname = cases(ci).name;
    r = store.(cname);
    iid = strcmp(conditions,'identity'); iaws = strcmp(conditions,'align_width_shoulder');
    mb = mean(r.acc_mean_per_seed(iid,:)) - r.chance;
    ma = mean(r.acc_mean_per_seed(iaws,:)) - r.chance;
    L{end+1} = sprintf('%s: above-chance margin %.3f unmodified -> %.3f after the fullest transform (%.0f%% retained).', ...
        cname, mb, ma, 100*ma/mb);
end
L{end+1} = '  This retained fraction is an UPPER BOUND on performance independent of the targeted properties.';
L{end+1} = '  Residual peak, FWHM and centroid variation survives every transform, so the retained accuracy may';
L{end+1} = '  still draw on the properties the transform was meant to remove. It CANNOT be apportioned to';
L{end+1} = '  unidentified spectral structure.';

results = store;
results.inference = struct('single', sing, 'single_p_holm', p_sing, ...
    'cumulative', cum, 'cumulative_p_holm', p_cum, 'ref_mode', opt.ref_mode);

save(fullfile(out_dir, sprintf('input_ablation_results%s.mat', tag)), '-struct', 'results');
fid = fopen(fullfile(out_dir, sprintf('input_ablation_summary%s.txt', tag)), 'w');
fprintf(fid, '%s\n', L{:});
fclose(fid);
fprintf('%s\n', L{:});

for ci = 1:numel(cases)
    cname = cases(ci).name;
    make_ablation_figure(conditions, store.(cname), cname, cases(ci).dyes, out_dir, tag);
end
fprintf('\nOutputs written to %s\n', out_dir);
end

% =====================================================================

function c = contrast(r, conditions, pr, cname, n_seeds)
ab = pr{1}; ct = pr{2}; prop = pr{3};
ia = strcmp(conditions, ab); ic = strcmp(conditions, ct);
d = r.acc_mean_per_seed(ic,:) - r.acc_mean_per_seed(ia,:);
md = mean(d); sd = std(d);
se = sd / sqrt(n_seeds);
tcrit = tinv(0.975, n_seeds-1);
tstat = md / max(se, eps);
pval = 2 * (1 - tcdf(abs(tstat), n_seeds-1));
c = struct('label', [cname '_' ab], 'case', cname, 'ablation', ab, 'control', ct, ...
    'property', prop, 'd', d, 'mean', md, 'ci_lo', md - tcrit*se, 'ci_hi', md + tcrit*se, 'p', pval);
end

function padj = holm(p)
% Holm-Bonferroni step-down adjusted p-values, monotonicity enforced.
n = numel(p);
[ps, ord] = sort(p);
adj = zeros(1,n);
running = 0;
for i = 1:n
    running = max(running, (n - i + 1) * ps(i));
    adj(i) = min(running, 1);
end
padj = zeros(1,n);
padj(ord) = adj;
end

function s = verdict_str(sig)
if sig
    s = 'transformation causes a measurable loss';
else
    s = 'not resolvable after correction';
end
end

function make_ablation_figure(conditions, r, cname, dyes, out_dir, tag)
mv = mean(r.acc_mean_per_seed, 2);
sv = std(r.acc_mean_per_seed, 0, 2);
is_ab = ismember(conditions, {'peak_align','centroid_align','width_norm','shoulder_remove', ...
                              'align_width','align_width_shoulder'});
disp_lab = strrep(conditions, '_', ' ');
figure('Position', [100 100 1150 560]);
b = bar(categorical(disp_lab, disp_lab), mv, 'FaceColor', 'flat');
for q = 1:numel(conditions)
    if is_ab(q), b.CData(q,:) = [0.85 0.33 0.10]; else, b.CData(q,:) = [0.30 0.50 0.75]; end
end
hold on;
errorbar(1:numel(conditions), mv, sv, 'k', 'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 8);
h1 = yline(mv(strcmp(conditions,'identity')), 'b--', 'LineWidth', 1.5);
h2 = yline(r.chance, 'k:', 'LineWidth', 1.5);
ylabel('CV accuracy (mean over runs)'); ylim([0 1]); grid on;
set(gca, 'TickLabelInterpreter', 'none');
legend([b h1 h2], {'orange = feature-targeting transform, blue = control', ...
    sprintf('baseline, unmodified (%.3f)', mv(strcmp(conditions,'identity'))), ...
    sprintf('chance (%.3f)', r.chance)}, 'Location', 'southwest');
title(sprintf('%s (BODIPY %s): accuracy after feature-targeting transforms; error bars = sd over %d runs', ...
    cname, strjoin(compose('%d', dyes), '/'), r.n_seeds), 'Interpreter', 'none');
save_figure(gcf, fullfile(out_dir, sprintf('input_ablation_%s%s', cname, tag)));
end

function net = train_one(X, y, n_classes, max_epochs)
sz = size(X);
lgraph = dual_branch_graph(sz, n_classes);
augmenter = imageDataAugmenter('RandXReflection', true, 'RandYReflection', false);
aug = augmentedImageDatastore([sz(1) sz(2) sz(3)], X, y, 'DataAugmentation', augmenter);
options = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', max_epochs, ...
    'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'none');
net = trainNetwork(aug, lgraph, options);
end

function lgraph = dual_branch_graph(sz, n_labels)
inputLayer = imageInputLayer([sz(1) sz(2) sz(3)], 'Name', 'input');
convLayers = [
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(32, 'Name', 'fc_conv')];
flattenInputLayer = [
    averagePooling2dLayer([1, 5], 'Name', 'gap_input')
    fullyConnectedLayer(32, 'Name', 'fc_input')];
concatLayer = concatenationLayer(1, 2, 'Name', 'concat');
finalLayers = [
    reluLayer('Name', 'relu_concat')
    fullyConnectedLayer(n_labels, 'Name', 'fc_final')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')];
lgraph = layerGraph(inputLayer);
lgraph = addLayers(lgraph, convLayers);
lgraph = addLayers(lgraph, flattenInputLayer);
lgraph = addLayers(lgraph, concatLayer);
lgraph = addLayers(lgraph, finalLayers);
lgraph = connectLayers(lgraph, 'input', 'conv1');
lgraph = connectLayers(lgraph, 'input', 'gap_input');
lgraph = connectLayers(lgraph, 'fc_conv', 'concat/in1');
lgraph = connectLayers(lgraph, 'fc_input', 'concat/in2');
lgraph = connectLayers(lgraph, 'concat', 'relu_concat');
end

function [psd, fsd, csd] = property_spread(O, wl)
n = size(O,3);
pk = nan(n,1); fw = nan(n,1); ct = nan(n,1);
for i = 1:n
    prof = movmean(mean(O(:,:,i),2),3);
    [pv, pr] = max(prof);
    pk(i) = wl(pr);
    base = min(prof); half = base + 0.5*(pv-base);
    l = find(prof(1:pr) <= half, 1, 'last');
    r = find(prof(pr:end) <= half, 1, 'first');
    if isempty(l), wl_l = wl(1); else, wl_l = wl(l); end
    if isempty(r), wl_r = wl(end); else, wl_r = wl(pr+r-1); end
    fw(i) = wl_r - wl_l;
    w = max(prof, 0);
    ct(i) = sum(wl .* w) / max(sum(w), eps);
end
psd = std(pk); fsd = std(fw); csd = std(ct);
end
