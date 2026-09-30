function occlusion_multiseed(varargin)
% OCCLUSION_MULTISEED
% Replicates the band-occlusion importance profile across independently
% trained networks, to test whether the profile is a property of the
% classification problem or an artifact of one particular initialization
% and fold.
%
% WHY THIS IS NEEDED. occlusion_band_importance.m reports a paired
% bootstrap confidence interval, which resamples held-out MOLECULES. It
% therefore says nothing about variability across TRAINING RUNS: the
% profile it measures belongs to one set of weights, produced by one
% random initialization on one fold. If the 602-637 nm result were an
% idiosyncrasy of that run, no statistic in that script could reveal it.
% This one retrains from scratch n_seeds times per case and reports the
% across-network distribution of the profile.
%
% Also extends the sweep across dye pairs spanning the observed difficulty
% range, including BODIPY-6 vs BODIPY-7, which classifies at chance under
% cross-validation (0.503 +/- 0.006). That pair is a TRAINED negative
% control and a stronger one than the random-initialization control in the
% original script: a network that has been trained normally but cannot
% discriminate should show a flat occlusion profile. If it instead shows
% structure, the sweep is measuring something other than discriminative
% dependence.

p = inputParser;
% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addParameter(p, 'base_dir', repo_root);
addParameter(p, 'n_seeds', 10);
addParameter(p, 'window_nm', 15);
addParameter(p, 'max_epochs', 200);
addParameter(p, 'replot_only', false);   % redraw figures from saved results, no retraining
parse(p, varargin{:});
opt = p.Results;

base_dir = opt.base_dir;
data_dir = fullfile(base_dir, 'data', 'processed_combined');
cutoff_val = 1000;
out_dir = fullfile(base_dir, 'results', 'interpretability', 'multiseed');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

% Difficulty spread taken from the saved CV results: 1-13 near the top,
% 4-10 at the median (the existing reference pair), 6-7 at chance.
cases = struct( ...
    'name',      {'triplet_2_11_13','pair_1_13','pair_4_10','pair_6_7'}, ...
    'dyes',      {[2 11 13],        [1 13],     [4 10],     [6 7]}, ...
    'n_per_dye', {610,              600,        600,        600}, ...
    'role',      {'primary result', 'high-accuracy pair', 'median pair (reference)', 'CHANCE-LEVEL PAIR: trained negative control'});

if opt.replot_only
    R = load(fullfile(out_dir, 'occlusion_multiseed_results.mat'));
    for ci = 1:numel(cases)
        cname = cases(ci).name;
        if ~isfield(R, cname), continue; end
        r = R.(cname);
        make_multiseed_figure(r.wl, r.dacc_all, r.mean_profile, r.sd_profile, cname, ...
            r.role, size(r.dacc_all,2), opt.window_nm, out_dir);
        fprintf('  %s: figures redrawn from saved results\n', cname);
    end
    return;
end

L = {};
L{end+1} = 'Multi-Seed Replication of the Band-Occlusion Importance Profile';
L{end+1} = '===============================================================';
L{end+1} = sprintf('%d independently trained networks per case (fresh initialization and fresh stratified fold each).', opt.n_seeds);
L{end+1} = sprintf('Occlusion window %g nm, replaced by the per-band population mean, as in occlusion_band_importance.m.', opt.window_nm);
L{end+1} = '';

results = struct();

for ci = 1:numel(cases)
    cname = cases(ci).name;
    dyes = cases(ci).dyes;
    n_per_dye = cases(ci).n_per_dye;
    n_classes = numel(dyes);
    fprintf('\n============ %s (BODIPY %s) ============\n', cname, mat2str(dyes));

    rng(1000 + ci, 'twister');

    % Load the full filtered population once; each seed draws its own
    % balanced subsample and its own fold, so the replication covers
    % sampling of molecules as well as of initializations.
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
    n_h = numel(wl);
    band_mean = mean(cell2mat(cellfun(@(r) mean(reshape(r, n_h, []), 2), pop', 'UniformOutput', false)), 2);
    win_rows = window_rows(wl, opt.window_nm);

    dacc_all = nan(n_h, opt.n_seeds);
    acc0_all = nan(opt.n_seeds, 1);
    peak_wl_all = nan(opt.n_seeds, 1);

    for s = 1:opt.n_seeds
        X = []; y = [];
        for k = 1:n_classes
            idx = randperm(size(pop{k},3), n_per_dye);
            X = cat(3, X, pop{k}(:,:,idx));
            y = [y; k*ones(n_per_dye,1)]; %#ok<AGROW>
        end
        y = categorical(y);
        X = reshape(X, [size(X,1), size(X,2), 1, size(X,3)]);
        cvp = cvpartition(y, 'KFold', 5, 'Stratify', true);
        tr = cvp.training(1); te = cvp.test(1);

        net = train_one(X(:,:,:,tr), y(tr), n_classes, opt.max_epochs);
        Xte = X(:,:,:,te); yte = double(y(te));
        c0 = classify(net, Xte) == y(te);
        acc0_all(s) = mean(c0);

        n_w = size(Xte,2); n_te = size(Xte,4);
        d = zeros(n_h,1);
        for i = 1:n_h
            rows = win_rows{i};
            Xo = Xte;
            Xo(rows,:,:,:) = repmat(band_mean(rows), [1 n_w 1 n_te]);
            d(i) = acc0_all(s) - mean(classify(net, Xo) == y(te));
        end
        dacc_all(:,s) = d;
        [~, im] = max(d);
        peak_wl_all(s) = wl(im);
        fprintf('  seed %2d: baseline acc %.3f, max drop %.3f at %.1f nm\n', s, acc0_all(s), max(d), wl(im));
    end

    mu = mean(dacc_all, 2);
    sd = std(dacc_all, 0, 2);
    [mx, imx] = max(mu);
    in_region = peak_wl_all >= 600 & peak_wl_all <= 640;

    L{end+1} = sprintf('=== %s: BODIPY %s (%s) ===', cname, strjoin(compose('%d', dyes), '/'), cases(ci).role);
    L{end+1} = sprintf('Baseline accuracy across %d networks: %.3f +/- %.3f (chance %.3f)', ...
        opt.n_seeds, mean(acc0_all), std(acc0_all), 1/n_classes);
    L{end+1} = sprintf('Mean profile peak: %.3f accuracy drop at %.1f nm (across-network SD at that window %.3f)', ...
        mx, wl(imx), sd(imx));
    L{end+1} = sprintf('Per-network peak location: median %.1f nm, range %.1f-%.1f nm', ...
        median(peak_wl_all), min(peak_wl_all), max(peak_wl_all));
    L{end+1} = sprintf('Networks whose peak falls in 600-640 nm: %d of %d', sum(in_region), opt.n_seeds);
    L{end+1} = sprintf('Largest drop per network: median %.3f, range %.3f-%.3f', ...
        median(max(dacc_all,[],1)), min(max(dacc_all,[],1)), max(max(dacc_all,[],1)));
    L{end+1} = '';

    results.(cname) = struct('dyes', dyes, 'wl', wl, 'dacc_all', dacc_all, ...
        'mean_profile', mu, 'sd_profile', sd, 'acc0', acc0_all, ...
        'peak_wl', peak_wl_all, 'role', cases(ci).role, 'n_classes', n_classes);

    make_multiseed_figure(wl, dacc_all, mu, sd, cname, cases(ci).role, opt.n_seeds, opt.window_nm, out_dir);
end

save(fullfile(out_dir, 'occlusion_multiseed_results.mat'), '-struct', 'results');
fid = fopen(fullfile(out_dir, 'occlusion_multiseed_summary.txt'), 'w');
fprintf(fid, '%s\n', L{:});
fclose(fid);
fprintf('%s\n', L{:});
fprintf('\nOutputs written to %s\n', out_dir);
end


function make_multiseed_figure(wl, dacc_all, mu, sd, cname, role, n_seeds, window_nm, out_dir)
figure('Position', [100 100 1000 520]); hold on;
fill([wl; flipud(wl)], [mu-sd; flipud(mu+sd)], [0.2 0.4 0.8], ...
    'FaceAlpha', 0.20, 'EdgeColor', 'none', 'DisplayName', 'across-network +/- 1 SD');
for s = 1:size(dacc_all,2)
    plot(wl, dacc_all(:,s), '-', 'Color', [0.6 0.6 0.6 0.5], 'LineWidth', 0.6, 'HandleVisibility', 'off');
end
plot(wl, mu, '-', 'LineWidth', 2.4, 'Color', [0.1 0.25 0.7], 'DisplayName', 'mean of networks');
yline(0, 'k--', 'HandleVisibility', 'off');
xlabel('Wavelength (nm)');
ylabel(sprintf('Drop in held-out accuracy (%g nm window occluded)', window_nm));
title(sprintf('%s: occlusion profile across %d independently trained networks (%s)', ...
    strrep(cname,'_','-'), n_seeds, role), 'Interpreter', 'none');
legend('Location','best'); grid on;
save_figure(gcf, fullfile(out_dir, sprintf('occlusion_multiseed_%s', cname)));
end

function rows = window_rows(wl, width_nm)
n = numel(wl); rows = cell(n,1);
for i = 1:n
    rows{i} = find(abs(wl - wl(i)) <= width_nm/2);
end
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
