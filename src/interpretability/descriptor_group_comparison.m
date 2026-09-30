function descriptor_group_comparison(varargin)
% DESCRIPTOR_GROUP_COMPARISON
% Replaces the three-group feature comparison in features_only_classifier.m
% with a five-group ontology and a dimensionality-matched comparison.
%
% WHY THE EARLIER GROUPING WAS INADEQUATE
%  1. Spectral centroid was placed in the "band shape" group because it is
%     a whole-band rather than a landmark measure. But the centroid is an
%     intensity-weighted mean WAVELENGTH -- a band-POSITION statistic. It
%     is also the single most predictive descriptor in both cases, so
%     filing it under shape made "band shape rather than peak position"
%     look far better supported than it is.
%  2. The groups had 6, 1 and 2 members, so the shape group's advantage
%     was confounded with dimensionality.
%
% This script therefore (a) separates band position (centroid) from
% width/shape, (b) reports every group at matched size by scoring each
% group's BEST SINGLE descriptor as well as the full group, and (c) runs an
% explicit dimensionality control in which the one-descriptor groups are
% padded to the largest group's size with pure noise.
%
% Matched-size comparison is the decisive one for the reviewer's question,
% because the reviewer named three properties, not three feature-set sizes.

p = inputParser;
% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addParameter(p, 'base_dir', repo_root);
addParameter(p, 'k_folds', 5);
addParameter(p, 'n_noise_reps', 20);
parse(p, varargin{:});
opt = p.Results;

base_dir = opt.base_dir;
data_dir = fullfile(base_dir, 'data', 'processed_combined');
cutoff_val = 1000;
out_dir = fullfile(base_dir, 'results', 'interpretability', 'ablation');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

rng(42, 'twister');

cases = struct('name', {'pairwise','triplet'}, ...
               'dyes', {[4 10], [2 11 13]}, ...
               'n_per_dye', {600, 610});

% Five groups. 'Band position' is separated from 'Width / shape' -- the
% distinction the earlier three-group version elided.
% Explicit struct keys rather than names derived from the labels: MATLAB's
% name-sanitizing helper does not turn 'Width / shape' into 'WidthShape'.
groups = struct( ...
    'label', {'Peak maximum', 'Band position', 'Width / shape', 'Tail', 'Shoulder'}, ...
    'key',   {'PeakMaximum',  'BandPosition',  'WidthShape',    'Tail', 'Shoulder'}, ...
    'feats', {{'peak_position','peak_intensity'}, ...
              {'spectral_centroid'}, ...
              {'fwhm','band_width_90','asymmetry','kurtosis'}, ...
              {'red_edge_integral'}, ...
              {'shoulder_intensity','shoulder_position'}});

L = {};
L{end+1} = 'Descriptor-Group Comparison under a Revised Ontology, with Dimensionality Matching';
L{end+1} = '=================================================================================';
L{end+1} = 'Spectral centroid is separated from width/shape descriptors: it is an intensity-weighted mean';
L{end+1} = 'wavelength, i.e. a band-POSITION statistic, and grouping it with shape overstated the case for';
L{end+1} = 'shape over position. Groups are compared both in full and at matched size (best single descriptor).';
L{end+1} = '';

results = struct();

for ci = 1:numel(cases)
    cname = cases(ci).name;
    dyes = cases(ci).dyes;
    n_per_dye = cases(ci).n_per_dye;

    profiles = []; labels = []; wl = [];
    for k = 1:numel(dyes)
        d = load(fullfile(data_dir, sprintf('bodipy%d.mat', dyes(k))));
        if isempty(wl), wl = d.wl(:); end
        sel = d.n_photons_combined > cutoff_val;
        r = d.spectral_rois_combined(:,:,sel);
        cen = d.spectral_centroids_combined(sel);
        [~, rm] = rmoutliers(cen, 'mean', 'thresholdfactor', 4);
        r = double(r(:,:,~rm));
        idx = randperm(size(r,3), n_per_dye);
        profiles = [profiles, squeeze(mean(r(:,:,idx), 2))]; %#ok<AGROW>
        labels = [labels; k*ones(n_per_dye,1)]; %#ok<AGROW>
    end
    y = categorical(labels);
    [X, names] = extract_spectral_features(profiles, wl);
    ok = all(isfinite(X), 2);
    X = X(ok,:); y = y(ok);
    cv = cvpartition(y, 'KFold', opt.k_folds, 'Stratify', true);
    chance = 1/numel(dyes);

    L{end+1} = sprintf('=== %s: BODIPY %s, n = %d, chance = %.3f ===', ...
        cname, strjoin(compose('%d', dyes), '/'), numel(y), chance);
    L{end+1} = '';
    L{end+1} = sprintf('  %-16s %-6s %-16s %-28s %s', 'Group', '#feat', 'full group', 'best single descriptor', 'name of best');
    max_size = max(arrayfun(@(g) numel(g.feats), groups));
    gres = struct();
    for g = 1:numel(groups)
        cols = find(ismember(names, groups(g).feats));
        full_acc = cv_acc(X(:,cols), y, cv);
        singles = arrayfun(@(c) cv_acc(X(:,c), y, cv), cols);
        [best, bi] = max(singles);
        L{end+1} = sprintf('  %-16s %-6d %-16.3f %-28.3f %s', groups(g).label, numel(cols), ...
            full_acc, best, names{cols(bi)});
        gres.(groups(g).key) = struct( ...
            'full', full_acc, 'best_single', best, 'best_name', names{cols(bi)}, ...
            'n_feat', numel(cols), 'singles', singles, 'feat_names', {names(cols)});
    end
    all_acc = cv_acc(X, y, cv);
    L{end+1} = sprintf('  %-16s %-6d %-16.3f', 'ALL descriptors', numel(names), all_acc);

    % Dimensionality control: pad each group to max_size with noise.
    L{end+1} = '';
    L{end+1} = sprintf('Dimensionality control -- each group padded to %d columns with pure Gaussian noise (%d repeats):', ...
        max_size, opt.n_noise_reps);
    for g = 1:numel(groups)
        cols = find(ismember(names, groups(g).feats));
        if numel(cols) >= max_size
            L{end+1} = sprintf('  %-16s (already %d columns, no padding needed)', groups(g).label, numel(cols));
            continue;
        end
        pad = max_size - numel(cols);
        reps = zeros(opt.n_noise_reps,1);
        for t = 1:opt.n_noise_reps
            reps(t) = cv_acc([X(:,cols), randn(size(X,1), pad)], y, cv);
        end
        L{end+1} = sprintf('  %-16s %.3f +/- %.3f   (vs %.3f unpadded)', groups(g).label, ...
            mean(reps), std(reps), gres.(groups(g).key).full);
    end

    % The matched-size verdict.
    bests = arrayfun(@(g) gres.(g.key).best_single, groups);
    [~, wi] = max(bests);
    L{end+1} = '';
    L{end+1} = sprintf('MATCHED-SIZE VERDICT: the strongest single descriptor of any group is %s (%s), at %.3f.', ...
        gres.(groups(wi).key).best_name, groups(wi).label, bests(wi));
    pm = gres.PeakMaximum.best_single;
    ws = gres.WidthShape.best_single;
    L{end+1} = sprintf('  Width/shape best single (%s, %.3f) vs peak-maximum best single (%s, %.3f): difference %+.3f', ...
        gres.WidthShape.best_name, ws, gres.PeakMaximum.best_name, pm, ws - pm);
    L{end+1} = sprintf('  Band position (centroid, %.3f) vs peak-maximum best single (%.3f): difference %+.3f', ...
        gres.BandPosition.best_single, pm, gres.BandPosition.best_single - pm);
    L{end+1} = '';

    results.(cname) = struct('dyes', dyes, 'groups', gres, 'all_descriptors', all_acc, ...
        'chance', chance, 'group_labels', {{groups.label}});

    % Figure
    figure('Position', [100 100 980 500]);
    fulls = arrayfun(@(g) gres.(g.key).full, groups);
    b = bar(categorical({groups.label}, {groups.label}), [fulls(:), bests(:)]);
    hold on;
    yline(all_acc, 'k-', 'LineWidth', 1.5);
    yline(chance, 'k:', 'LineWidth', 1.5);
    ylabel('5-fold CV accuracy'); ylim([0 1]); grid on;
    legend([b(1) b(2)], {'full group', 'best single descriptor (matched size)'}, 'Location', 'best');
    title(sprintf('%s (BODIPY %s): descriptor groups, full vs. matched size', ...
        cname, strjoin(compose('%d', dyes), '/')));
    save_figure(gcf, fullfile(out_dir, sprintf('descriptor_groups_%s', cname)));
end

save(fullfile(out_dir, 'descriptor_group_results.mat'), '-struct', 'results');
fid = fopen(fullfile(out_dir, 'descriptor_group_summary.txt'), 'w');
fprintf(fid, '%s\n', L{:});
fclose(fid);
fprintf('%s\n', L{:});
fprintf('\nOutputs written to %s\n', out_dir);
end

function a = cv_acc(X, y, cv)
k = cv.NumTestSets;
fa = zeros(k,1);
for i = 1:k
    tr = cv.training(i); te = cv.test(i);
    m = fitcdiscr(X(tr,:), y(tr), 'DiscrimType', 'pseudolinear');
    fa(i) = mean(predict(m, X(te,:)) == y(te));
end
a = mean(fa);
end
