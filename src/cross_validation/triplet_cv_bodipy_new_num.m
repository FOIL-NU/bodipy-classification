function triplet_cv_bodipy_new_num()
% TRIPLET_CV_BODIPY_NEW_NUM
% Stratified 5-fold cross-validation for the six representative BODIPY
% dye triads that bodipy_new_num already treats individually with
% dedicated scripts: analyze_bodipy_triplets_{1_3_9,1_4_13,1_6_13,
% 2_11_13,3_5_13,3_13_15}.m. This supersedes the previous draft of this
% validation study, which ran against the yang/bsa BSA-conjugated
% B4/B6/B13 triplet -- a different, now-superseded experiment.
%
% Each triad's own preprocessing (photon cutoff 1000, centroid outlier
% removal) and balanced per-dye sample count are reproduced from its
% source script. Filter counts, however, all use conv1=16/conv2=32 --
% the canonical dual-branch architecture in analyze_bodipy_pairs.m
% (manuscript Fig. 4A) and analyze_bodipy_triplets2.m, whose output
% folder is literally named triplet/cnn16_32/ and which sweeps all 364
% triad combinations (almost certainly the source of manuscript Fig.
% 4D/E and S60/S61). An earlier version of this script instead used
% each individual triad script's own filter counts verbatim, which for
% four of the six triads (1-3-9, 2-11-13, 3-5-13, 3-13-15) meant the
% smaller conv1=8/conv2=16 variant -- an earlier/exploratory
% architecture, not the one that matches the manuscript's systematic
% sweep. Triad [1,3,9]'s source script additionally used an older
% single-branch/20-epoch/LR-0.01 design predating the dual-branch/200-
% epoch approach entirely; it is trained here with the same
% dual-branch/200-epoch/LR=0.001/16-32-filter methodology as the other
% five triads for consistency with the canonical architecture.
%
% Reports mean +/- std of macro accuracy/precision/recall/F1 (F1 is the
% genuine mean of per-class F1, not the harmonic mean of macro P/R) and
% per-class precision/recall across folds. Checkpoints after every
% triad. Saves a representative trained net (triad 2-11-13, the only
% one with a pre-existing confusion matrix / figure export in the
% current pipeline, i.e. the one most likely already in the manuscript)
% for interpretability analysis.


% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
rng(42, 'twister');
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

data_dir = fullfile('data', 'processed_combined');
cutoff_val = 1000;
k_folds = 5;

% [dye1 dye2 dye3 n_per_dye conv1_filters conv2_filters]
% Sample sizes match each triad's own source script; filter counts are
% all conv1=16/conv2=32, the canonical architecture (see header).
triad_configs = {
    [1,  3,  9],  500, 16, 32;
    [1,  4,  13], 600, 16, 32;
    [1,  6,  13], 700, 16, 32;
    [2,  11, 13], 610, 16, 32;
    [3,  5,  13], 700, 16, 32;
    [3,  13, 15], 700, 16, 32;
};
n_triads = size(triad_configs, 1);
reference_triad = [2, 11, 13];

out_dir = fullfile(repo_root, 'results', 'cross_validation');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
results_file = fullfile(out_dir, 'triplet_cv_results.mat');

all_results = cell(n_triads, 1);
t_start = tic;

for t = 1:n_triads
    dyes = triad_configs{t, 1};
    n_per_dye = triad_configs{t, 2};
    conv1_f = triad_configs{t, 3};
    conv2_f = triad_configs{t, 4};

    fprintf('\n[%d/%d] Triad %d-%d-%d (n=%d/dye, elapsed %.1f min)\n', ...
        t, n_triads, dyes(1), dyes(2), dyes(3), n_per_dye, toc(t_start)/60);

    class_rois = cell(1,3);
    ok = true;
    for c = 1:3
        [rois, dye_ok] = load_and_filter(data_dir, dyes(c), cutoff_val);
        if ~dye_ok || size(rois,3) < n_per_dye
            fprintf('  Skipping: dye %d has insufficient samples\n', dyes(c));
            ok = false;
            break;
        end
        class_rois{c} = rois;
    end

    r = struct('dyes', dyes, 'n_per_dye', n_per_dye, 'skipped', ~ok);
    if ~ok
        all_results{t} = r;
        save(results_file, 'all_results');
        continue;
    end

    sampled = cell(1,3);
    for c = 1:3
        idx = randperm(size(class_rois{c},3), n_per_dye);
        sampled{c} = class_rois{c}(:,:,idx);
    end

    all_data = cat(3, sampled{1}, sampled{2}, sampled{3});
    all_labels = categorical([ones(n_per_dye,1); 2*ones(n_per_dye,1); 3*ones(n_per_dye,1)]);
    sz3 = size(all_data);
    all_data = reshape(all_data, [sz3(1), sz3(2), 1, sz3(3)]);

    cv = cvpartition(all_labels, 'KFold', k_folds, 'Stratify', true);
    fold_metrics = zeros(k_folds, 10); % acc, prec_macro, rec_macro, f1_macro, prec1-3, rec1-3

    for fold = 1:k_folds
        [net, val_data_f, val_labels_f] = train_triplet_fold(all_data, all_labels, cv, fold, conv1_f, conv2_f);
        val_pred = classify(net, val_data_f);
        cm = confusionmat(val_labels_f, val_pred);
        [acc, prec_macro, rec_macro, f1_macro, prec_c, rec_c] = calculate_metrics_full(cm);
        fold_metrics(fold, :) = [acc, prec_macro, rec_macro, f1_macro, prec_c, rec_c];
        fprintf('    Fold %d/%d: Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f\n', ...
            fold, k_folds, acc, prec_macro, rec_macro, f1_macro);
    end

    r.mean = mean(fold_metrics, 1);
    r.std = std(fold_metrics, 1);
    r.all_fold_metrics = fold_metrics;
    r.metric_names = {'Accuracy','Precision_macro','Recall_macro','F1_macro', ...
        sprintf('Precision_B%d',dyes(1)), sprintf('Precision_B%d',dyes(2)), sprintf('Precision_B%d',dyes(3)), ...
        sprintf('Recall_B%d',dyes(1)), sprintf('Recall_B%d',dyes(2)), sprintf('Recall_B%d',dyes(3))};

    fprintf('  Triad %d-%d-%d: Acc=%.3f+/-%.3f Prec=%.3f+/-%.3f Rec=%.3f+/-%.3f F1=%.3f+/-%.3f\n', ...
        dyes(1), dyes(2), dyes(3), r.mean(1), r.std(1), r.mean(2), r.std(2), r.mean(3), r.std(3), r.mean(4), r.std(4));

    all_results{t} = r;
    save(results_file, 'all_results');

    if isequal(sort(dyes), sort(reference_triad))
        % Reuse the last fold's trained net + its real held-out ROI
        % images/labels (no retraining needed).
        val_idx = cv.test(k_folds);
        roi_data = all_data(:,:,:,val_idx);
        roi_labels = all_labels(val_idx);
        save_reference_net(net, dyes, data_dir, roi_data, roi_labels, out_dir);
    end
end

fprintf('\nAll triads complete in %.1f minutes.\n', toc(t_start)/60);
write_summary(out_dir, all_results, cutoff_val);
% Raw per-fold metrics alongside the summary, so the reported mean +/- std can
% be independently recomputed rather than taken on faith.
write_folds_csv(out_dir, all_results);

end

function write_folds_csv(out_dir, all_results)
% Writes the per-fold metrics underlying triplet_cv_summary.txt. Column order
% matches r.metric_names: [Accuracy, Precision_macro, Recall_macro, F1_macro,
%  Precision_c1..c3, Recall_c1..c3].
csv_path = fullfile(out_dir, 'triplet_cv_folds.csv');
fid = fopen(csv_path, 'w');
if fid < 0
    warning('Could not open %s for writing; per-fold CSV not saved.', csv_path);
    return;
end
fprintf(fid, ['dye1,dye2,dye3,fold,accuracy,precision_macro,recall_macro,f1_macro,' ...
    'precision_dye1,precision_dye2,precision_dye3,recall_dye1,recall_dye2,recall_dye3\n']);
for i = 1:numel(all_results)
    r = all_results{i};
    if ~isfield(r, 'all_fold_metrics') || isempty(r.all_fold_metrics) || r.skipped
        continue;
    end
    M = r.all_fold_metrics;
    for f = 1:size(M, 1)
        fprintf(fid, '%d,%d,%d,%d', r.dyes(1), r.dyes(2), r.dyes(3), f);
        fprintf(fid, ',%.6f', M(f, :));
        fprintf(fid, '\n');
    end
end
fclose(fid);
fprintf('Per-fold metrics written to %s\n', csv_path);
end

function [rois, ok] = load_and_filter(data_dir, dye_id, cutoff_val)
    fpath = fullfile(data_dir, sprintf('bodipy%d.mat', dye_id));
    if ~exist(fpath, 'file')
        rois = [];
        ok = false;
        return;
    end
    d = load(fpath);
    sel = d.n_photons_combined > cutoff_val;
    rois = d.spectral_rois_combined(:,:,sel);
    centroid = d.spectral_centroids_combined(sel);
    [~, rm] = rmoutliers(centroid, 'mean', 'thresholdfactor', 4);
    rois = rois(:,:,~rm);
    ok = true;
end

function [net, val_data, val_labels] = train_triplet_fold(all_data, all_labels, cv, fold, conv1_f, conv2_f)
    train_idx = cv.training(fold);
    val_idx = cv.test(fold);

    train_data = all_data(:,:,:,train_idx);
    train_labels = all_labels(train_idx);
    val_data_raw = all_data(:,:,:,val_idx);
    val_labels = all_labels(val_idx);

    sz = size(train_data);
    lgraph = triplet_layer_graph(sz, conv1_f, conv2_f, numel(categories(all_labels)));

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

function lgraph = triplet_layer_graph(sz, conv1_f, conv2_f, n_labels)
    % Dual-branch architecture, matching bodipy_new_num's
    % analyze_bodipy_triplets_*.m scripts (filter counts vary by triad).
    inputLayer = imageInputLayer([sz(1) sz(2) sz(3)], 'Name', 'input');

    convLayers = [
        convolution2dLayer(3, conv1_f, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        convolution2dLayer(3, conv2_f, 'Padding', 'same', 'Name', 'conv2')
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

function [accuracy, precision, recall, f1score, precision_per_class, recall_per_class] = calculate_metrics_full(confusion_matrix)
    % Genuine macro-F1: the mean of each class's own F1 score, NOT the
    % harmonic mean of macro-averaged precision and recall (those two
    % quantities differ whenever per-class precision/recall diverge --
    % confirmed to matter here: for triads 1-4-13 and 1-6-13, the old
    % formula overstated F1 by 0.028-0.032 relative to this one).
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

function save_reference_net(net, dyes, data_dir, roi_data, roi_labels, out_dir)
    % Saves the trained net together with the REAL held-out ROI images
    % it was validated on (roi_data/roi_labels), so cnn_weight_analysis.m
    % can compute genuine forward-pass activations rather than a
    % synthetic proxy.
    d1 = load(fullfile(data_dir, sprintf('bodipy%d.mat', dyes(1))));
    wl = d1.wl; %#ok<NASGU>
    reference_triad = dyes; %#ok<NASGU>
    net_file = fullfile(out_dir, 'triplet_reference_net.mat');
    save(net_file, 'net', 'wl', 'reference_triad', 'roi_data', 'roi_labels');
    fprintf('Reference net saved to %s (with %d real held-out ROI images)\n', net_file, size(roi_data,4));
end

function write_summary(out_dir, all_results, cutoff_val)
    fid = fopen(fullfile(out_dir, 'triplet_cv_summary.txt'), 'w');
    fprintf(fid, 'Triplet CNN Classification (bodipy_new_num) - Stratified 5-Fold CV Results\n');
    fprintf(fid, '==========================================================================\n\n');
    fprintf(fid, 'Photon cutoff: %d\n\n', cutoff_val);
    for i = 1:length(all_results)
        r = all_results{i};
        if isempty(r)
            continue;
        end
        fprintf(fid, 'Triad %d-%d-%d (n=%d/dye):\n', r.dyes(1), r.dyes(2), r.dyes(3), r.n_per_dye);
        if r.skipped
            fprintf(fid, '  SKIPPED (insufficient samples)\n\n');
            continue;
        end
        fprintf(fid, '  Accuracy:  %.3f +/- %.3f\n', r.mean(1), r.std(1));
        fprintf(fid, '  Precision: %.3f +/- %.3f\n', r.mean(2), r.std(2));
        fprintf(fid, '  Recall:    %.3f +/- %.3f\n', r.mean(3), r.std(3));
        fprintf(fid, '  F1 Score:  %.3f +/- %.3f\n', r.mean(4), r.std(4));
        fprintf(fid, '  Per-class:\n');
        for c = 1:3
            fprintf(fid, '    BODIPY-%d: Precision=%.3f+/-%.3f  Recall=%.3f+/-%.3f\n', ...
                r.dyes(c), r.mean(4+c), r.std(4+c), r.mean(7+c), r.std(7+c));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    fprintf('Summary written to %s\n', fullfile(out_dir, 'triplet_cv_summary.txt'));
end
