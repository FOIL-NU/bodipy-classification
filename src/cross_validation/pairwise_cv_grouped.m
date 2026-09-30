function pairwise_cv_grouped(pairs_to_test)
% PAIRWISE_CV_GROUPED
% Acquisition-file-grouped cross-validation, as a check on whether the
% ordinary per-molecule stratified 5-fold CV in
% pairwise_cv_bodipy_new_num.m is optimistic due to same-acquisition
% correlations (molecules from the same imaging session/field-of-view
% sharing instrument/environment conditions could appear in both the
% train and validation fold of a purely per-molecule split).
%
% Reconstructs which raw acquisition file (from
% data/upstream/old_processed_files/, indexed via index.xlsx and
% read_indexfile.m) each molecule in processed_combined/bodipy*.mat
% came from, by replaying the same concatenation order as
% step3_combine_files.m. Folds are then built by holding out whole
% acquisition files, not individual molecules.
%
% Sanity check built in: the reconstructed per-file molecule counts
% must sum EXACTLY to the combined array length, or grouping is
% considered unavailable for that dye and this falls back to a
% single group (i.e. no real grouping -- flagged explicitly in the
% output, not silently treated as if grouping succeeded).
%
% Known limitation: dyes 3, 9, and 12 have no matching row in
% index.xlsx at all (a pre-existing data indexing gap, not something
% this script can resolve) -- any pair involving them cannot be
% grouped and is reported as such.
%
% Because each dye only has 3-8 acquisition files, this uses however
% many groups the smaller dye in each pair actually has (capped at 5,
% floored at 2 -- pairs where the smaller dye has <2 usable files are
% skipped), rather than forcing 5 folds.
%
% Usage: pairwise_cv_grouped({[1,13],[6,7],[2,13],[1,2],[3,5]})


% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
rng(42, 'twister');
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end
addpath(fullfile(repo_root, 'data', 'upstream')); % read_indexfile.mlx lives here

data_dir = fullfile('data', 'processed_combined');
old_files_dir = fullfile('data', 'upstream', 'old_processed_files');
index_file = fullfile('data', 'upstream', 'index.xlsx');
cutoff_val = 1000;
n_per_dye = 600;

out_dir = fullfile(repo_root, 'results', 'cross_validation');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
results_file = fullfile(out_dir, 'pairwise_grouped_cv_results.mat');

n_pairs = length(pairs_to_test);
all_results = cell(n_pairs, 1);
t_start = tic;

for p = 1:n_pairs
    dye1 = pairs_to_test{p}(1);
    dye2 = pairs_to_test{p}(2);
    fprintf('\n[%d/%d] BODIPY %d vs %d (elapsed %.1f min)\n', p, n_pairs, dye1, dye2, toc(t_start)/60);

    [rois1, gid1, ok1, hasgrp1, nfiles1] = load_with_groups(data_dir, old_files_dir, index_file, dye1, cutoff_val);
    [rois2, gid2, ok2, hasgrp2, nfiles2] = load_with_groups(data_dir, old_files_dir, index_file, dye2, cutoff_val);

    r = struct('dye1', dye1, 'dye2', dye2, 'has_groups', hasgrp1 && hasgrp2, ...
        'n_files_dye1', nfiles1, 'n_files_dye2', nfiles2, 'skipped', false);

    if ~ok1 || ~ok2 || size(rois1,3) < n_per_dye || size(rois2,3) < n_per_dye
        fprintf('  Skipping: insufficient samples\n');
        r.skipped = true;
        all_results{p} = r;
        save(results_file, 'all_results');
        continue;
    end

    if ~r.has_groups
        fprintf('  No acquisition-file grouping available for this pair (dye %d groups=%d, dye %d groups=%d) -- skipping grouped CV\n', ...
            dye1, nfiles1, dye2, nfiles2);
        r.skipped = true;
        all_results{p} = r;
        save(results_file, 'all_results');
        continue;
    end

    k_groups = max(2, min([5, nfiles1, nfiles2]));
    fprintf('  Using %d acquisition-file-based folds (dye%d has %d files, dye%d has %d files)\n', ...
        k_groups, dye1, nfiles1, dye2, nfiles2);

    % Subsample to n_per_dye/dye, but keep group ids aligned.
    i_idx = randperm(size(rois1,3), n_per_dye);
    j_idx = randperm(size(rois2,3), n_per_dye);
    rois1 = rois1(:,:,i_idx); gid1 = gid1(i_idx);
    rois2 = rois2(:,:,j_idx); gid2 = gid2(j_idx);

    all_data = cat(3, rois1, rois2);
    all_labels = categorical([zeros(n_per_dye,1); ones(n_per_dye,1)]);
    sz3 = size(all_data);
    all_data = reshape(all_data, [sz3(1), sz3(2), 1, sz3(3)]);

    % Assign each of the two dyes' acquisition files to one of k_groups
    % folds (round-robin over shuffled file order), independently per
    % dye so both classes contribute to every fold.
    fold_of_file1 = assign_files_to_folds(unique(gid1), k_groups);
    fold_of_file2 = assign_files_to_folds(unique(gid2), k_groups);
    fold_id = zeros(size(all_data,4), 1);
    fold_id(1:n_per_dye) = fold_of_file1(gid1);
    fold_id(n_per_dye+1:end) = fold_of_file2(gid2);

    fold_metrics = nan(k_groups, 4);
    for fold = 1:k_groups
        val_idx = (fold_id == fold);
        train_idx = ~val_idx;
        if sum(val_idx) < 4 || numel(unique(all_labels(val_idx))) < 2
            fprintf('    Fold %d/%d: skipped (too few / single-class validation samples)\n', fold, k_groups);
            continue;
        end
        [net, val_data_f, val_labels_f] = train_grouped_fold(all_data, all_labels, train_idx, val_idx);
        val_pred = classify(net, val_data_f);
        cm = confusionmat(val_labels_f, val_pred);
        [acc, prec, rec, f1] = calculate_metrics_full(cm);
        fold_metrics(fold, :) = [acc, prec, rec, f1];
        fprintf('    Fold %d/%d (held-out files, n_val=%d): Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f\n', ...
            fold, k_groups, sum(val_idx), acc, prec, rec, f1);
    end

    valid_folds = ~any(isnan(fold_metrics), 2);
    r.k_groups = k_groups;
    r.mean = mean(fold_metrics(valid_folds,:), 1);
    r.std = std(fold_metrics(valid_folds,:), 1);
    r.n_valid_folds = sum(valid_folds);
    r.all_fold_metrics = fold_metrics;

    fprintf('  Pair %d-%d (grouped, %d/%d usable folds): Acc=%.3f+/-%.3f F1=%.3f+/-%.3f\n', ...
        dye1, dye2, r.n_valid_folds, k_groups, r.mean(1), r.std(1), r.mean(4), r.std(4));

    all_results{p} = r;
    save(results_file, 'all_results');
end

fprintf('\nGrouped CV complete in %.1f minutes.\n', toc(t_start)/60);
write_grouped_summary(out_dir, all_results);

end

function [rois, group_id, ok, has_groups, n_files] = load_with_groups(data_dir, old_files_dir, index_file, dye_id, cutoff_val)
    fpath = fullfile(data_dir, sprintf('bodipy%d.mat', dye_id));
    if ~exist(fpath, 'file')
        rois = []; group_id = []; ok = false; has_groups = false; n_files = 0;
        return;
    end
    d = load(fpath);
    n_total = length(d.n_photons_combined);

    has_groups = false;
    n_files = 0;
    group_id_full = ones(n_total, 1);
    try
        filenames = read_indexfile(index_file, dye_id);
        n_files = length(filenames);
        gid = zeros(n_total, 1);
        running = 0;
        for jj = 1:n_files
            fd = load(fullfile(old_files_dir, [lower(char(filenames(jj))) '.mat']), 'spectral_profiles');
            n_mol = size(fd.spectral_profiles, 2);
            gid(running+1:running+n_mol) = jj;
            running = running + n_mol;
        end
        if running == n_total
            group_id_full = gid;
            has_groups = true;
        else
            fprintf('  [dye %d] group reconstruction mismatch (%d vs %d molecules) -- grouping unavailable\n', ...
                dye_id, running, n_total);
        end
    catch
        % index.xlsx has no row for this dye (known gap for dyes 3, 9, 12) -- grouping unavailable.
    end

    sel = d.n_photons_combined > cutoff_val;
    rois = d.spectral_rois_combined(:,:,sel);
    centroid = d.spectral_centroids_combined(sel);
    [~, rm] = rmoutliers(centroid, 'mean', 'thresholdfactor', 4);
    rois = rois(:,:,~rm);
    gid_sel = group_id_full(sel);
    group_id = gid_sel(~rm);
    ok = true;
end

function fold_of_file = assign_files_to_folds(file_ids, k_groups)
    file_ids = file_ids(randperm(length(file_ids)));
    fold_of_file = zeros(max(file_ids), 1);
    for k = 1:length(file_ids)
        fold_of_file(file_ids(k)) = mod(k-1, k_groups) + 1;
    end
end

function [net, val_data, val_labels] = train_grouped_fold(all_data, all_labels, train_idx, val_idx)
    train_data = all_data(:,:,:,train_idx);
    train_labels = all_labels(train_idx);
    val_data_raw = all_data(:,:,:,val_idx);
    val_labels = all_labels(val_idx);

    sz = size(train_data);
    lgraph = pairwise_layer_graph(sz, numel(categories(all_labels)));

    augmenter = imageDataAugmenter('RandXReflection', true, 'RandYReflection', false);
    aug_train = augmentedImageDatastore([sz(1) sz(2) sz(3)], train_data, train_labels, 'DataAugmentation', augmenter);
    val_data = augmentedImageDatastore([sz(1) sz(2) sz(3)], val_data_raw, val_labels);

    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.001, ...
        'MaxEpochs', 200, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', val_data, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'none');

    net = trainNetwork(aug_train, lgraph, options);
end

function lgraph = pairwise_layer_graph(sz, n_labels)
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

function [accuracy, precision, recall, f1score] = calculate_metrics_full(confusion_matrix)
    num_classes = size(confusion_matrix, 1);
    precision_per_class = zeros(1, num_classes);
    recall_per_class = zeros(1, num_classes);
    f1_per_class = zeros(1, num_classes);
    for c = 1:num_classes
        TP = confusion_matrix(c, c);
        FP = sum(confusion_matrix(:, c)) - TP;
        FN = sum(confusion_matrix(c, :)) - TP;
        if (TP + FP) > 0
            precision_per_class(c) = TP / (TP + FP);
        end
        if (TP + FN) > 0
            recall_per_class(c) = TP / (TP + FN);
        end
        if (precision_per_class(c) + recall_per_class(c)) > 0
            f1_per_class(c) = 2 * precision_per_class(c) * recall_per_class(c) / (precision_per_class(c) + recall_per_class(c));
        end
    end
    precision = mean(precision_per_class);
    recall = mean(recall_per_class);
    f1score = mean(f1_per_class);
    accuracy = trace(confusion_matrix) / sum(confusion_matrix(:));
end

function write_grouped_summary(out_dir, all_results)
    fid = fopen(fullfile(out_dir, 'pairwise_grouped_cv_summary.txt'), 'w');
    fprintf(fid, 'Pairwise CNN Classification - Acquisition-File-Grouped CV\n');
    fprintf(fid, '===========================================================\n\n');
    fprintf(fid, 'Folds are held-out ACQUISITION FILES, not individual molecules.\n');
    fprintf(fid, 'k = min(5, min(n_files_dye1, n_files_dye2)), floor 2.\n\n');
    for i = 1:length(all_results)
        r = all_results{i};
        if isempty(r); continue; end
        if r.skipped
            fprintf(fid, '%2d-%-2d   SKIPPED (grouping unavailable or insufficient samples; n_files=%d/%d)\n', ...
                r.dye1, r.dye2, r.n_files_dye1, r.n_files_dye2);
            continue;
        end
        fprintf(fid, '%2d-%-2d   k=%d (%d usable)   Acc=%.3f+/-%.3f   Prec=%.3f+/-%.3f   Rec=%.3f+/-%.3f   F1=%.3f+/-%.3f\n', ...
            r.dye1, r.dye2, r.k_groups, r.n_valid_folds, r.mean(1), r.std(1), r.mean(2), r.std(2), r.mean(3), r.std(3), r.mean(4), r.std(4));
    end
    fclose(fid);
    fprintf('Summary written to %s\n', fullfile(out_dir, 'pairwise_grouped_cv_summary.txt'));
end
