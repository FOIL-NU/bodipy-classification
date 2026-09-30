function smoke_test()
% SMOKE_TEST
% Quick end-to-end sanity check: confirms the data files, MATLAB
% toolboxes, and CNN pipeline in this repository all work on your
% machine, in under a minute -- before committing to the ~2-2.5 hour
% full pairwise CV run in Section 5.2 of the README. This is NOT a
% meaningful classification result (it trains for 3 epochs on 40
% molecules/dye); it only checks that every step runs without error and
% produces output of the expected shape.
%
% Runs as a function, not a script, specifically so it does not clear or
% otherwise touch your MATLAB base workspace.

repo_root = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(repo_root, 'data', 'processed_combined');

fprintf('=== BODIPY CNN pipeline smoke test ===\n\n');
n_checks = 0;
n_passed = 0;

%% 1. Data file exists and has the expected fields/shapes
n_checks = n_checks + 1;
fprintf('[1] Loading data/processed_combined/bodipy1.mat ... ');
fpath = fullfile(data_dir, 'bodipy1.mat');
assert(exist(fpath, 'file') == 2, 'bodipy1.mat not found at %s', fpath);
d1 = load(fpath);
required_fields = {'wl', 'spectral_rois_combined', 'spectral_profiles_combined', ...
    'n_photons_combined', 'spectral_centroids_combined'};
for f = 1:numel(required_fields)
    assert(isfield(d1, required_fields{f}), 'Missing expected field: %s', required_fields{f});
end
n_wl = numel(d1.wl);
n_mol = size(d1.spectral_rois_combined, 3);
assert(size(d1.spectral_rois_combined, 1) == n_wl, 'spectral_rois_combined dim 1 should match numel(wl)');
assert(size(d1.spectral_rois_combined, 2) == 5, 'spectral_rois_combined dim 2 should be 5 (ROI width)');
assert(numel(d1.n_photons_combined) == n_mol, 'n_photons_combined length should match ROI count');
fprintf('OK (%d molecules, %d-point wavelength grid)\n', n_mol, n_wl);
n_passed = n_passed + 1;

%% 2. Second dye file loads too (need at least 2 dyes for pairwise classification)
n_checks = n_checks + 1;
fprintf('[2] Loading data/processed_combined/bodipy2.mat ... ');
d2 = load(fullfile(data_dir, 'bodipy2.mat'));
fprintf('OK (%d molecules)\n', size(d2.spectral_rois_combined, 3));
n_passed = n_passed + 1;

%% 3. Preprocessing (photon-count filter + centroid outlier removal) runs and leaves enough molecules
n_checks = n_checks + 1;
fprintf('[3] Preprocessing (photon-count filter + outlier removal) ... ');
n_per_dye = 40;
cutoff_val = 1000;
[rois1, ok1] = load_and_filter_smoke(d1, cutoff_val);
[rois2, ok2] = load_and_filter_smoke(d2, cutoff_val);
assert(ok1 && ok2, 'Preprocessing failed for one of the two dyes');
assert(size(rois1,3) >= n_per_dye && size(rois2,3) >= n_per_dye, ...
    'Fewer than %d molecules survive preprocessing (dye1=%d, dye2=%d) -- cannot run the smoke test, but this alone does not mean the full pipeline is broken (dyes 1/2 comfortably exceed this in the full dataset)', ...
    n_per_dye, size(rois1,3), size(rois2,3));
fprintf('OK (dye1: %d -> usable, dye2: %d -> usable)\n', size(rois1,3), size(rois2,3));
n_passed = n_passed + 1;

%% 4. Dual-branch CNN builds, trains for a few epochs, and classifies -- on a tiny subset
n_checks = n_checks + 1;
fprintf('[4] Building + training the dual-branch CNN on a tiny subset (this may take ~10-30s) ... ');
rng(42, 'twister');
i_idx = randperm(size(rois1,3), n_per_dye);
j_idx = randperm(size(rois2,3), n_per_dye);
all_data = cat(3, rois1(:,:,i_idx), rois2(:,:,j_idx));
all_labels = categorical([zeros(n_per_dye,1); ones(n_per_dye,1)]);
sz3 = size(all_data);
all_data = reshape(all_data, [sz3(1), sz3(2), 1, sz3(3)]);

cv = cvpartition(all_labels, 'HoldOut', 0.3, 'Stratify', true);
train_idx = cv.training;
test_idx = cv.test;

sz = size(all_data(:,:,:,train_idx));
lgraph = smoke_test_layer_graph(sz, numel(categories(all_labels)));

train_ds = augmentedImageDatastore([sz(1) sz(2) sz(3)], all_data(:,:,:,train_idx), all_labels(train_idx));
test_ds = augmentedImageDatastore([sz(1) sz(2) sz(3)], all_data(:,:,:,test_idx), all_labels(test_idx));

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 3, ...
    'Verbose', false, ...
    'Plots', 'none');

net = trainNetwork(train_ds, lgraph, options);
pred = classify(net, test_ds);
assert(numel(pred) == sum(test_idx), 'classify() returned the wrong number of predictions');
acc = mean(pred == all_labels(test_idx));
fprintf('OK (ran to completion; %d-epoch/%d-sample smoke accuracy=%.2f, not meaningful)\n', ...
    options.MaxEpochs, n_per_dye*2, acc);
n_passed = n_passed + 1;

%% 5. extract_spectral_features.m runs on a real spectrum (used by the interpretability scripts)
n_checks = n_checks + 1;
fprintf('[5] extract_spectral_features.m on a real held-out spectrum ... ');
addpath(fullfile(repo_root, 'src', 'interpretability'));
profile1 = squeeze(mean(all_data(:,:,1,test_idx), 2)); % [n_wl x n_test], average across the 5-pixel ROI width
[feat, feat_names] = extract_spectral_features(profile1, d1.wl);
assert(size(feat,1) == sum(test_idx), 'extract_spectral_features returned the wrong number of rows');
assert(size(feat,2) == numel(feat_names), 'feature matrix width should match number of feature names');
fprintf('OK (%d features extracted per spectrum)\n', numel(feat_names));
n_passed = n_passed + 1;

fprintf('\n=== %d/%d checks passed ===\n', n_passed, n_checks);
if n_passed == n_checks
    fprintf('Your MATLAB installation, toolboxes, and the data files in this repository are all working correctly.\n');
else
    error('smoke_test:failed', '%d/%d checks failed -- see the output above for the first failure.', n_checks - n_passed, n_checks);
end

end

function [rois, ok] = load_and_filter_smoke(d, cutoff_val)
    sel = d.n_photons_combined > cutoff_val;
    rois = d.spectral_rois_combined(:,:,sel);
    centroid = d.spectral_centroids_combined(sel);
    [~, rm] = rmoutliers(centroid, 'mean', 'thresholdfactor', 4);
    rois = rois(:,:,~rm);
    ok = true;
end

function lgraph = smoke_test_layer_graph(sz, n_labels)
    % Same dual-branch architecture as pairwise_cv_bodipy_new_num.m /
    % analyze_bodipy_pairs.m (manuscript Fig. 4A) -- see those files for
    % the full annotated version used in the actual reported results.
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
