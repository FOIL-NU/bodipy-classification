function features_only_classifier(varargin)
% FEATURES_ONLY_CLASSIFIER
% Trains conventional classifiers on the 10 explicit spectral features
% from extract_spectral_features.m (Approach 1), under the SAME dyes,
% preprocessing, sample sizes and stratified 5-fold CV scheme as the
% CNN, and compares:
%
%   (a) features-only accuracy vs. CNN accuracy -- how much of the CNN's
%       performance is reachable from named, human-readable features at
%       all;
%   (b) FEATURE-GROUP accuracy, with the groups chosen to match the
%       three candidates the reviewer named -- peak position, shoulder
%       intensity, overall band shape -- so the answer is phrased in the
%       reviewer's own vocabulary rather than in ours;
%   (c) single-feature and drop-one-feature importance within the full
%       feature set.
%
% This complements occlusion_band_importance.m rather than repeating it.
% Occlusion says WHERE ON THE WAVELENGTH AXIS the trained CNN's decision
% lives; this says WHICH NAMED SPECTRAL PROPERTY carries class
% information in the data. Agreement between the two is what licenses a
% statement like "the CNN uses band shape" instead of "the CNN attends
% near 611 nm".
%
% IMPORTANT SCOPE LIMIT, stated up front: a features-only classifier is
% evidence about what the DATA supports, not proof of what the CNN
% internally computes. A feature group that classifies well need not be
% the group the CNN uses, and one that classifies poorly could still be
% used by the CNN in combination with others. The claim this analysis
% can support is the conjunction: a feature group both (i) carries
% class information on its own here, and (ii) sits where the CNN's
% occlusion profile is largest. Neither leg alone is sufficient.
%
% Usage:
%   features_only_classifier
%   features_only_classifier('k_folds', 5)

p = inputParser;
addParameter(p, 'k_folds', 5);
% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addParameter(p, 'base_dir', repo_root);
parse(p, varargin{:});
opt = p.Results;

base_dir = opt.base_dir;
data_dir = fullfile(base_dir, 'data', 'processed_combined');
cutoff_val = 1000;
out_dir = fullfile(base_dir, 'results', 'interpretability', 'occlusion');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

rng(42, 'twister');   % same seed the CV scripts use

% Matches save_reference_net / triad_configs in the CV scripts.
cases = struct( ...
    'name',      {'pairwise', 'triplet'}, ...
    'dyes',      {[4 10],     [2 11 13]}, ...
    'n_per_dye', {600,        610});

% Reviewer's three named candidates, mapped onto the 10 extracted
% features. 'Band shape' deliberately collects every width/asymmetry/
% tail descriptor, since that is what the reviewer's phrase "overall
% band shape" denotes; peak INTENSITY is grouped with amplitude rather
% than with peak POSITION, because conflating them is exactly the
% confound the activation analysis fell into.
groups = struct( ...
    'label', {'Peak position', 'Shoulder', 'Band shape', 'Amplitude', 'All 10 features'}, ...
    'feats', {{'peak_position'}, ...
              {'shoulder_intensity','shoulder_position'}, ...
              {'fwhm','band_width_90','asymmetry','kurtosis','red_edge_integral','spectral_centroid'}, ...
              {'peak_intensity'}, ...
              {}});

L = {};
L{end+1} = 'Features-Only Classifiers vs. the CNN, by Named Spectral Feature Group';
L{end+1} = '=======================================================================';
L{end+1} = sprintf('Stratified %d-fold CV, same dyes / photon cutoff (%d) / outlier filtering / per-dye sample sizes as the CNN validation.', ...
    opt.k_folds, cutoff_val);
L{end+1} = 'Classifiers: linear discriminant (primary) and bagged trees (guards against understating a';
L{end+1} = 'feature group that is informative only nonlinearly). Both are reported; neither is tuned.';
L{end+1} = '';

results = struct();

for ci = 1:numel(cases)
    cname = cases(ci).name;
    dyes = cases(ci).dyes;
    n_per_dye = cases(ci).n_per_dye;
    fprintf('\n=== %s: dyes %s ===\n', cname, mat2str(dyes));

    % --- Load, filter and balance exactly as the CV scripts do ---
    profiles = [];
    labels = [];
    wl = [];
    for k = 1:numel(dyes)
        f = fullfile(data_dir, sprintf('bodipy%d.mat', dyes(k)));
        d = load(f);
        if isempty(wl), wl = d.wl(:); end
        sel = d.n_photons_combined > cutoff_val;
        rois = d.spectral_rois_combined(:,:,sel);
        centroid = d.spectral_centroids_combined(sel);
        [~, rm] = rmoutliers(centroid, 'mean', 'thresholdfactor', 4);
        rois = double(rois(:,:,~rm));
        if size(rois,3) < n_per_dye
            error('Dye %d has only %d molecules after filtering, need %d.', dyes(k), size(rois,3), n_per_dye);
        end
        idx = randperm(size(rois,3), n_per_dye);
        rois = rois(:,:,idx);
        profiles = [profiles, squeeze(mean(rois, 2))]; %#ok<AGROW>  [H x n]
        labels = [labels; k*ones(n_per_dye,1)];        %#ok<AGROW>
    end
    y = categorical(labels);

    % --- Extract the 10 named features ---
    [X, feat_names] = extract_spectral_features(profiles, wl);
    bad = ~all(isfinite(X), 2);
    if any(bad)
        L{end+1} = sprintf('NOTE (%s): %d of %d molecules dropped for non-finite feature values.', ...
            cname, sum(bad), numel(bad));
        X = X(~bad, :); y = y(~bad);
    end

    cv = cvpartition(y, 'KFold', opt.k_folds, 'Stratify', true);

    L{end+1} = sprintf('=== %s network: BODIPY %s (n = %d molecules) ===', cname, ...
        strjoin(arrayfun(@(d) sprintf('%d', d), dyes, 'UniformOutput', false), '/'), numel(y));

    % --- CNN reference accuracy, read from the saved CV results ---
    [cnn_acc, cnn_sd, cnn_note] = cnn_reference_accuracy(base_dir, cname, dyes);
    L{end+1} = sprintf('CNN reference: %s', cnn_note);

    % --- Feature-group comparison ---
    L{end+1} = '';
    L{end+1} = 'Accuracy by named feature group (mean +/- std over folds):';
    L{end+1} = sprintf('  %-18s %-6s %-22s %-22s', 'Group', '#feat', 'linear discriminant', 'bagged trees');
    grp_acc = nan(numel(groups), 2);
    for g = 1:numel(groups)
        if isempty(groups(g).feats)
            cols = 1:numel(feat_names);
        else
            cols = find(ismember(feat_names, groups(g).feats));
        end
        [a1, s1] = cv_accuracy(X(:,cols), y, cv, 'lda');
        [a2, s2] = cv_accuracy(X(:,cols), y, cv, 'bag');
        grp_acc(g,:) = [a1 a2];
        L{end+1} = sprintf('  %-18s %-6d %.3f +/- %.3f          %.3f +/- %.3f', ...
            groups(g).label, numel(cols), a1, s1, a2, s2);
    end
    chance = 1/numel(dyes);
    L{end+1} = sprintf('  (chance = %.3f)', chance);

    % --- Per-feature importance within the full set ---
    L{end+1} = '';
    L{end+1} = 'Per-feature importance within the full 10-feature set (linear discriminant):';
    L{end+1} = sprintf('  %-22s %-14s %-14s', 'Feature', 'alone', 'drop-one loss');
    single_acc = nan(numel(feat_names),1);
    drop_loss  = nan(numel(feat_names),1);
    [full_acc, ~] = cv_accuracy(X, y, cv, 'lda');
    for f = 1:numel(feat_names)
        single_acc(f) = cv_accuracy(X(:,f), y, cv, 'lda');
        keep = setdiff(1:numel(feat_names), f);
        drop_loss(f) = full_acc - cv_accuracy(X(:,keep), y, cv, 'lda');
    end
    [~, ord] = sort(single_acc, 'descend');
    for f = ord(:)'
        L{end+1} = sprintf('  %-22s %.3f          %+.4f', feat_names{f}, single_acc(f), drop_loss(f));
    end
    L{end+1} = sprintf('  Full 10-feature accuracy: %.3f', full_acc);

    % --- Read-out against the reviewer's question ---
    [best_grp_acc, best_grp] = max(grp_acc(1:end-1, 1));   % exclude the 'all features' row
    L{end+1} = '';
    L{end+1} = sprintf('Best single named group (linear): %s at %.3f, vs %.3f for all 10 features and %.3f for the CNN.', ...
        groups(best_grp).label, best_grp_acc, grp_acc(end,1), cnn_acc);
    gap = cnn_acc - grp_acc(end,1);
    L{end+1} = sprintf('CNN minus all-10-features gap: %+.3f accuracy%s', gap, ...
        ternary_str(abs(gap) < 2*max(cnn_sd, 0.01), ...
            ' -- within about 2 CNN CV standard deviations, i.e. the named features account for most of what the CNN achieves.', ...
            ' -- outside 2 CNN CV standard deviations, i.e. the CNN uses information these 10 named features do not capture.'));
    L{end+1} = '';

    results.(cname) = struct('dyes', dyes, 'feat_names', {feat_names}, ...
        'group_labels', {{groups.label}}, 'group_accuracy_lda_bag', grp_acc, ...
        'single_feature_accuracy', single_acc, 'drop_one_loss', drop_loss, ...
        'full_feature_accuracy', full_acc, 'cnn_accuracy', cnn_acc, 'cnn_sd', cnn_sd, ...
        'chance', chance);

    % --- Figure ---
    figure('Position', [100 100 900 480]);
    b = bar(categorical({groups.label}, {groups.label}), grp_acc);
    hold on;
    yline(cnn_acc, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('CNN 5-fold CV (%.3f)', cnn_acc));
    yline(chance, 'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('chance (%.3f)', chance));
    legend([b(1) b(2) findobj(gca,'Type','ConstantLine')'], ...
        {'linear discriminant', 'bagged trees', sprintf('chance (%.3f)', chance), sprintf('CNN (%.3f)', cnn_acc)}, ...
        'Location', 'best');
    ylabel('5-fold CV accuracy'); ylim([0 1]); grid on;
    title(sprintf('%s (BODIPY %s): accuracy from named spectral feature groups vs. the CNN', ...
        cname, strjoin(arrayfun(@(d) sprintf('%d', d), dyes, 'UniformOutput', false), '/')));
    save_figure(gcf, fullfile(out_dir, sprintf('feature_group_accuracy_%s', cname)));
end

save(fullfile(out_dir, 'features_only_classifier_results.mat'), '-struct', 'results');
fid = fopen(fullfile(out_dir, 'features_only_classifier_summary.txt'), 'w');
fprintf(fid, '%s\n', L{:});
fclose(fid);
fprintf('%s\n', L{:});
fprintf('\nOutputs written to %s\n', out_dir);

end

% =====================================================================

function [acc, sd] = cv_accuracy(X, y, cv, kind)
k = cv.NumTestSets;
fold_acc = zeros(k,1);
for i = 1:k
    tr = cv.training(i); te = cv.test(i);
    switch kind
        case 'lda'
            % 'pseudolinear' rather than 'linear': single-feature and
            % small-group models routinely produce a singular pooled
            % covariance, which plain 'linear' errors on.
            mdl = fitcdiscr(X(tr,:), y(tr), 'DiscrimType', 'pseudolinear');
        case 'bag'
            mdl = fitcensemble(X(tr,:), y(tr), 'Method', 'Bag', 'NumLearningCycles', 100);
    end
    fold_acc(i) = mean(predict(mdl, X(te,:)) == y(te));
end
acc = mean(fold_acc);
sd = std(fold_acc);
end

function [acc, sd, note] = cnn_reference_accuracy(base_dir, cname, dyes)
% Reads the already-computed CNN 5-fold CV accuracy for the same dyes
% out of validation_results, rather than re-quoting a number by hand.
acc = NaN; sd = NaN;
try
    if strcmp(cname, 'pairwise')
        d = load(fullfile(base_dir, 'results', 'cross_validation', 'pairwise_cv_results.mat'));
        R = d.all_results;
        for i = 1:numel(R)
            if isequal(sort([R(i).dye1 R(i).dye2]), sort(dyes))
                acc = R(i).mean_accuracy;
                if isfield(R(i), 'std_accuracy'), sd = R(i).std_accuracy; end
                break;
            end
        end
    else
        d = load(fullfile(base_dir, 'results', 'cross_validation', 'triplet_cv_results.mat'));
        R = d.all_results;
        for i = 1:numel(R)
            ri = R{i};
            if isfield(ri, 'dyes') && isequal(sort(ri.dyes(:)'), sort(dyes))
                % Triad results store a metric VECTOR plus metric_names,
                % not scalar mean_accuracy/std_accuracy fields.
                mi = find(strcmp(ri.metric_names, 'Accuracy'), 1);
                acc = ri.mean(mi);
                sd  = ri.std(mi);
                break;
            end
        end
    end
catch ME
    note = sprintf('could not read saved CNN CV result (%s)', ME.message);
    return;
end
if isnan(acc)
    note = 'no matching entry found in the saved CV results';
else
    note = sprintf('5-fold CV accuracy %.3f +/- %.3f (from results/cross_validation, same dyes and preprocessing)', acc, sd);
end
end

function s = ternary_str(cond, a, b)
if cond, s = a; else, s = b; end
end
