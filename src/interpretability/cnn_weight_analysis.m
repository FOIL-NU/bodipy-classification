function cnn_weight_analysis(net_file, varargin)
% CNN_WEIGHT_ANALYSIS
% Interpretability analysis of a trained BODIPY CNN classifier.
%
% Unlike an earlier draft of this analysis, this version is designed to
% run only against a network trained on real BODIPY spectral data (see
% pairwise_cv_bodipy_new_num.m / triplet_cv_bodipy_new_num.m, which each
% save a representative trained net + the real ROI images it was
% trained/validated on). Running it against a network trained on
% synthetic data would demonstrate the method, not say anything about
% what the actual classifier learned -- so net_file must supply real
% roi_data/roi_labels, not a placeholder.
%
% It also replaces a "filter response to wavelength" analysis that
% previously convolved each filter with a synthetic unit impulse (which
% just replots the filter's raw weights, not a genuine response to
% data) with real forward-pass activations computed on the actual held
% -out spectra.
%
% IMPORTANT: peak activation magnitude is NOT the same thing as
% class-discriminative importance. A filter can be strongly active on
% every input regardless of class. This version separately computes,
% per filter: (1) whether its peak activation coincides with a
% zero-padded 'same'-convolution boundary row (row 1 or the last row --
% those two rows are the only ones affected by padding for a 3x3
% kernel; everything else is computed from real neighboring data), and
% (2) an actual class-separation score (Cohen's d for 2 classes, eta^2
% from a one-way ANOVA for 3+), reported alongside -- and explicitly
% NOT the same ranking as -- the raw peak-activation ranking. Checked
% empirically: for both the pairwise and triplet reference networks,
% the two rankings are close to uncorrelated (e.g. the single most
% class-separating filter was the *weakest* by peak activation, and the
% strongest-activation filter was middling for class separation).
% "Most active" and "most discriminative" are reported as two distinct
% things throughout, not conflated into one "most important" ranking.
%
% Inputs:
%   net_file  - path to .mat file containing 'net', 'wl', 'roi_data',
%               'roi_labels' (as saved by save_reference_net in the CV
%               scripts)
%   varargin  - optional: 'output_dir' (default
%               repo_root/results/interpretability/figures)

p = inputParser;
% Resolve the default output directory relative to this file
% (repo_root/src/interpretability/<this file>), so it does not depend on
% MATLAB's current working directory. An explicitly supplied relative
% 'output_dir' is still resolved against the caller's pwd, as usual.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addParameter(p, 'output_dir', fullfile(repo_root, 'results', 'interpretability', 'figures'));
parse(p, varargin{:});
output_dir = p.Results.output_dir;
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

d = load(net_file);
required = {'net', 'wl', 'roi_data', 'roi_labels'};
for r = required
    if ~isfield(d, r{1})
        error('cnn_weight_analysis:missingField', ...
            '%s is missing required field ''%s''. This function requires a net trained on real spectral data (see save_reference_net in the CV scripts), not a synthetic demo.', net_file, r{1});
    end
end
net = d.net;
wl = d.wl(:);
roi_data = d.roi_data;     % [H, W, 1, N] real ROI images
roi_labels = d.roi_labels; % categorical, length N

layers = net.Layers;

% --- First convolutional layer: raw filter weights ---
conv_idx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.Convolution2DLayer'), layers), 1, 'first');
if isempty(conv_idx)
    error('Could not find a convolutional layer in this network.');
end
conv1 = layers(conv_idx);
conv1_name = conv1.Name;
filters = conv1.Weights; % [fh, fw, channels, n_filters]
n_filters = size(filters, 4);

fprintf('First conv layer (''%s''): %d filters, size %dx%dx%d\n', ...
    conv1_name, n_filters, size(filters,1), size(filters,2), size(filters,3));

figure('Position', [100, 100, 1400, 800]);
n_cols = min(8, ceil(sqrt(n_filters)));
n_rows = ceil(n_filters / n_cols);
for f = 1:n_filters
    subplot(n_rows, n_cols, f);
    filt = squeeze(filters(:,:,1,f));
    filt = (filt - min(filt(:))) / (max(filt(:)) - min(filt(:)) + eps);
    imagesc(filt);
    axis image;
    colormap(gca, 'gray');
    title(sprintf('Filter %d', f));
    colorbar;
end
sgtitle('First-Layer Convolutional Filters (real trained network)');
save_figure(gcf, fullfile(output_dir, 'cnn_filters'));

% --- Real forward-pass activations on the actual training/validation data ---
fprintf('\nComputing real conv1 activations on %d held-out ROI images...\n', size(roi_data,4));
act = activations(net, roi_data, conv1_name); % [H, W, n_filters, N] ('same' padding => H matches input)

% Average across the ROI's narrow spatial axis (dim 2) and across
% samples (dim 4) to get each filter's mean activation as a function of
% wavelength (dim 1).
mean_act_per_wl = squeeze(mean(mean(act, 2), 4)); % [H x n_filters]
if size(mean_act_per_wl, 1) ~= length(wl)
    mean_act_per_wl = mean_act_per_wl'; % defensive transpose if dims came out swapped
end

avg_spectrum = squeeze(mean(mean(roi_data, 2), 4)); % [H x 1], real average spectrum

figure;
plot(wl, avg_spectrum, 'LineWidth', 2, 'Color', 'k');
xlabel('Wavelength (nm)'); ylabel('Mean spectral intensity (background-subtracted detector counts, a.u.)');
title('Average Spectral Profile (real held-out data)');
grid on;
save_figure(gcf, fullfile(output_dir, 'average_spectrum'));

figure('Position', [100, 100, 1200, 800]);
for f = 1:n_filters
    subplot(n_rows, n_cols, f);
    plot(wl, mean_act_per_wl(:, f), 'LineWidth', 2);
    xlabel('Wavelength (nm)'); ylabel('Mean activation');
    title(sprintf('Filter %d', f));
    grid on;
end
sgtitle('Filter Activations vs. Wavelength (real forward pass on held-out data)');
save_figure(gcf, fullfile(output_dir, 'filter_wavelength_responses'));

[peak_vals, peak_idx] = max(mean_act_per_wl, [], 1);
[~, ranked] = sort(peak_vals, 'descend');
fprintf('\nFilters ranked by peak real activation:\n');
for i = 1:n_filters
    f = ranked(i);
    fprintf('  Filter %d: peak activation %.4f at %.1f nm\n', f, peak_vals(f), wl(peak_idx(f)));
end

% --- Boundary-sensitivity check ---
% 'same' padding only pads the FIRST and LAST row for a 3x3 kernel;
% every interior row is computed entirely from real neighboring data.
% Flag any filter whose peak activation location changes when those two
% boundary rows are excluded.
n_h = size(mean_act_per_wl, 1);
interior_rows = 2:(n_h-1);
boundary_sensitive = false(n_filters, 1);
interior_peak_wl = zeros(n_filters, 1);
for f = 1:n_filters
    [~, interior_idx] = max(mean_act_per_wl(interior_rows, f));
    interior_peak_wl(f) = wl(interior_rows(interior_idx));
    boundary_sensitive(f) = (peak_idx(f) == 1) || (peak_idx(f) == n_h);
end
fprintf('\nBoundary-sensitivity check (peak at the zero-padded row 1 or row %d):\n', n_h);
for f = 1:n_filters
    if boundary_sensitive(f)
        fprintf('  Filter %d: PEAK AT PADDED BOUNDARY (row %d) -- interior peak instead at %.1f nm\n', ...
            f, peak_idx(f), interior_peak_wl(f));
    end
end
if ~any(boundary_sensitive)
    fprintf('  None -- all filters'' peak activations are at interior (non-padded) wavelengths.\n');
end

% --- Class-separation score, kept explicitly SEPARATE from activation magnitude ---
% Per-sample scalar summary per filter (mean activation over the whole
% spatial map), then Cohen's d (2 classes) or eta^2 from a one-way ANOVA
% (3+ classes) between classes. This answers "which filters actually
% distinguish the classes", which is a different question from "which
% filters are most active" (peak_vals above).
per_sample = reshape(squeeze(mean(mean(act, 1), 2)), n_filters, []);
labels_num = double(roi_labels);
n_classes_sep = numel(categories(roi_labels));
separation_score = zeros(n_filters, 1);
if n_classes_sep == 2
    c0 = per_sample(:, labels_num == 1);
    c1 = per_sample(:, labels_num == 2);
    pooled_std = sqrt((var(c0,0,2) + var(c1,0,2)) / 2);
    separation_score = abs(mean(c0,2) - mean(c1,2)) ./ pooled_std;
    separation_metric_name = 'Cohen''s d';
else
    for f = 1:n_filters
        [~, tbl] = anova1(per_sample(f,:), labels_num, 'off');
        separation_score(f) = tbl{2,2} / tbl{4,2}; % eta^2 = SS_between / SS_total
    end
    separation_metric_name = 'eta^2';
end
[~, rank_by_separation] = sort(separation_score, 'descend');
fprintf('\nFilters ranked by class-separation (%s) -- NOTE this is a different ranking from peak activation above:\n', separation_metric_name);
for i = 1:n_filters
    f = rank_by_separation(i);
    activation_rank = find(ranked == f);
    fprintf('  Filter %d: %s=%.4f  (peak-activation rank was %d of %d)\n', ...
        f, separation_metric_name, separation_score(f), activation_rank, n_filters);
end
rank_corr = corr(separation_score, peak_vals(:));
fprintf('  Correlation between activation magnitude and class-separation across filters: r=%.3f\n', rank_corr);

corr_matrix = corr(mean_act_per_wl, avg_spectrum);
figure;
bar(corr_matrix);
xlabel('Filter'); ylabel('Correlation with average spectrum (real data)');
title('Filter-Spectrum Correlation (real forward pass)');
grid on;
save_figure(gcf, fullfile(output_dir, 'filter_spectrum_correlation'));

% --- Per-class mean activation, to see which filters separate classes ---
class_names = categories(roi_labels);
n_classes = numel(class_names);
class_colors = lines(n_classes); % fixed palette, reused explicitly everywhere a
                                  % class needs to be color-matched across axes/plots
                                  % (yyaxis mode does NOT auto-cycle colors per line)
class_mean_act = zeros(size(mean_act_per_wl,1), n_filters, n_classes);
class_avg_spectrum = zeros(size(mean_act_per_wl,1), n_classes);
for c = 1:n_classes
    sel = roi_labels == class_names{c};
    class_mean_act(:,:,c) = squeeze(mean(mean(act(:,:,:,sel), 2), 4));
    class_avg_spectrum(:,c) = squeeze(mean(mean(roi_data(:,:,:,sel), 2), 4));
end
if n_classes >= 2
    figure('Position', [100, 100, 1200, 800]);
    for f = 1:n_filters
        subplot(n_rows, n_cols, f);
        hold on;
        for c = 1:n_classes
            plot(wl, class_mean_act(:,f,c), 'LineWidth', 1.5, 'DisplayName', class_names{c});
        end
        title(sprintf('Filter %d', f));
        if f == 1
            legend('Location','best');
        end
        grid on;
    end
    sgtitle('Per-Class Filter Activations (real forward pass)');
    save_figure(gcf, fullfile(output_dir, 'filter_per_class_activation'));
end

% --- Classification (final) layer weights ---
fc_idx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.FullyConnectedLayer'), layers), 1, 'last');
if ~isempty(fc_idx)
    fc = layers(fc_idx);
    weights = fc.Weights;
    biases = fc.Bias;
    fprintf('\nClassification layer (''%s''): weights %s, biases %s\n', ...
        fc.Name, mat2str(size(weights)), mat2str(biases'));

    figure;
    imagesc(weights);
    colorbar;
    xlabel('Input features'); ylabel('Output classes');
    yticks(1:n_classes); yticklabels(class_names);
    title('Classification Layer Weights');
    save_figure(gcf, fullfile(output_dir, 'classification_weights'));
end

% --- Direct-path (Branch 2) LOCAL LINEAR APPROXIMATION of effective weight vs. wavelength ---
% fc_final's 64 inputs are the concatenation of fc_conv (Branch 1: two
% conv+maxpool stages -- nonlinear, no clean per-wavelength mapping)
% and fc_input (Branch 2: avgpool(1x5) then a single FC layer -- purely
% linear in the 55 wavelength-averaged input values).
%
% Two SEPARATE claims are made below, with different strength:
%   (1) EXACT: which of the 64 concatenated inputs belong to fc_input
%       vs fc_conv is verified empirically (by matching real
%       activations, not assumed from the layer definition, since
%       concatenationLayer's dimension argument does not guarantee a
%       simple first-32/last-32 channel split once flattened into the
%       next FC layer). This mapping is exact -- either it is
%       recovered with zero residual or the whole decomposition below
%       is skipped.
%   (2) APPROXIMATE: composing fc_input's weights with fc_final's
%       corresponding weights treats the intervening ReLU as the
%       identity function. This is a local linear approximation of
%       Branch 2's contribution, not an exact decomposition of it --
%       the ReLU genuinely clips some units on some inputs, and this
%       composition does not account for that. It is labeled as such
%       throughout the output (never "exact weight composition"),
%       to avoid overstating what is, at best, a first-order
%       approximation valid where the ReLU stays in its linear regime.
direct_path_done = false;
if any(strcmp({layers.Name}, 'fc_input')) && any(strcmp({layers.Name}, 'fc_conv')) && any(strcmp({layers.Name}, 'fc_final'))
    fc_input_layer = layers(strcmp({layers.Name}, 'fc_input'));
    fc_conv_layer = layers(strcmp({layers.Name}, 'fc_conv'));
    fc_final_layer = layers(strcmp({layers.Name}, 'fc_final'));

    act_concat = activations(net, roi_data, 'concat');
    act_fc_input = activations(net, roi_data, 'fc_input');
    act_fc_conv = activations(net, roi_data, 'fc_conv');

    n_hold = size(roi_data, 4);
    concat_flat = reshape(act_concat, [], n_hold);   % 64 x N
    input_flat = reshape(act_fc_input, [], n_hold);  % 32 x N
    conv_flat = reshape(act_fc_conv, [], n_hold);    % 32 x N

    n_concat = size(concat_flat, 1);
    source_is_input = false(n_concat, 1);
    source_index = zeros(n_concat, 1);
    match_error = zeros(n_concat, 1);
    for k = 1:n_concat
        row = concat_flat(k, :);
        [err_in, idx_in] = min(sum((input_flat - row).^2, 2));
        [err_cv, idx_cv] = min(sum((conv_flat - row).^2, 2));
        if err_in <= err_cv
            source_is_input(k) = true;
            source_index(k) = idx_in;
            match_error(k) = err_in;
        else
            source_index(k) = idx_cv;
            match_error(k) = err_cv;
        end
    end

    if max(match_error) > 1e-6 * max(1, mean(concat_flat(:).^2))
        warning('cnn_weight_analysis:concatMappingUncertain', ...
            ['Could not confidently verify the concat-layer channel mapping for this network ' ...
             '(max match residual %.3g) -- skipping the direct-path wavelength decomposition rather ' ...
             'than report a possibly-wrong composition.'], max(match_error));
    elseif sum(source_is_input) ~= 32
        warning('cnn_weight_analysis:concatMappingUncertain', ...
            'Expected exactly 32 concat channels to map to fc_input, found %d -- skipping.', sum(source_is_input));
    else
        % fc_final columns (in fc_input's own 1..32 order) that receive Branch 2's output.
        fc_final_input_cols = zeros(size(fc_final_layer.Weights, 1), 32);
        input_positions = find(source_is_input);
        for j = 1:32
            k = input_positions(source_index(input_positions) == j);
            fc_final_input_cols(:, j) = fc_final_layer.Weights(:, k);
        end
        % Compose: [n_labels x 32] * [32 x 55] = [n_labels x 55].
        % (Treats the intermediate ReLU as locally linear -- an
        % approximation, not an exact decomposition.)
        direct_path_weight = fc_final_input_cols * fc_input_layer.Weights;

        figure('Position', [100, 100, 900, 500]);
        hold on;
        for c = 1:n_classes
            plot(wl, direct_path_weight(c, :), 'LineWidth', 2, 'DisplayName', class_names{c});
        end
        xlabel('Wavelength (nm)'); ylabel('Effective weight (direct path only, linear approximation)');
        title('Direct-Path (Branch 2) Effective Weight vs. Wavelength (Local Linear Approximation)');
        legend('Location', 'best');
        grid on;
        save_figure(gcf, fullfile(output_dir, 'direct_path_weight_vs_wavelength'));
        direct_path_done = true;

        fprintf('\nDirect-path (Branch 2) local linear approximation composed (concat-channel mapping verified exactly, max residual %.3g; the ReLU-as-identity composition itself is an approximation, not exact).\n', max(match_error));
    end
else
    fprintf('\nSkipping direct-path decomposition: this network does not have fc_conv/fc_input/fc_final layers (not the expected dual-branch architecture).\n');
end

% --- Occlusion-sensitivity vs. wavelength (covers BOTH branches, no linearity assumption) ---
% Unlike the weight-composition above, this handles the nonlinear
% conv+maxpool branch correctly: for each class c, systematically
% occlude one full wavelength row (all 5 spatial columns) at a time and
% measure the resulting change in class c's score, averaged over
% held-out samples WHOSE TRUE LABEL IS c (not all held-out samples --
% averaging class c's score-change over other classes' examples would
% be a valid "global score sensitivity" map, but is a different
% question from "what does the network look at in real examples of
% class c", which is what this is meant to answer and what the
% write-up describes it as). Positive = occluding that wavelength
% reduces confidence in the class (supporting evidence for it);
% negative = occluding it increases confidence (evidence against it).
%
% Occluded rows are replaced with occlusionSensitivity's 'auto'
% MaskValue, which is NOT zero -- verified empirically (by comparing
% against an explicit MaskValue=0 call) to equal the occluded image's
% own mean intensity. This is a smaller, less out-of-distribution
% perturbation than zeroing a full row, though still a substantial one
% (it removes all structure from that row, not just its scale).
fprintf('\nComputing occlusion sensitivity vs wavelength (%d classes, using only each class''s own held-out samples)...\n', n_classes);
occlusion_map = zeros(n_h, n_classes);
for c = 1:n_classes
    sel = find(roi_labels == class_names{c});
    per_sample_maps = zeros(n_h, length(sel));
    for si = 1:length(sel)
        per_sample_maps(:, si) = occlusionSensitivity(net, roi_data(:,:,:,sel(si)), class_names(c), ...
            'MaskSize', [1 size(roi_data,2)], 'Stride', [1 size(roi_data,2)], 'OutputUpsampling', 'none');
    end
    occlusion_map(:, c) = mean(per_sample_maps, 2);
    fprintf('  Class %s done (%d true-class samples).\n', class_names{c}, length(sel));
end

figure('Position', [100, 100, 900, 500]);
yyaxis left;
hold on;
for c = 1:n_classes
    plot(wl, occlusion_map(:, c), 'LineWidth', 2, 'Color', class_colors(c,:), 'LineStyle', '-', 'DisplayName', class_names{c});
end
yline(0, 'k--', 'HandleVisibility', 'off');
ax = gca; ax.YAxis(1).Color = 'k';
xlabel('Wavelength (nm)'); ylabel('Mean occlusion sensitivity (score drop when occluded)');
overlay_class_spectra(wl, class_avg_spectrum, n_classes, class_colors);
title('Occlusion Sensitivity vs. Wavelength, with Average Spectra Overlaid (dotted)');
yyaxis left;
legend('Location', 'best');
grid on;
save_figure(gcf, fullfile(output_dir, 'occlusion_sensitivity_vs_wavelength'));

% --- Input-gradient saliency vs. wavelength (third, complementary check) ---
% Occlusion sensitivity asks "what happens if this wavelength is
% entirely removed" (a large perturbation). Gradient saliency instead
% asks "how does the score change for an infinitesimal increase in
% this wavelength's value" (a local linear sensitivity), computed by
% backpropagating through the whole network -- including the
% max-pooling layers, which dlgradient handles correctly (no linearity
% assumption, unlike the direct-path decomposition above). Conditioned
% on true-label samples per class, same as occlusion sensitivity above,
% so the correlation between the two methods (reported below) compares
% the same population rather than class-c gradients over all samples
% against class-c occlusion over only class-c samples.
fprintf('\nComputing input-gradient saliency vs wavelength (%d classes, using only each class''s own held-out samples, matching the occlusion-sensitivity conditioning above)...\n', n_classes);
grad_map = zeros(n_h, n_classes);
grad_absmean_map = zeros(n_h, n_classes);
grad_logit_absmean_map = zeros(n_h, n_classes);
try
    cls_out_idx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.ClassificationOutputLayer'), layers), 1);
    lgraph_dl = layerGraph(net);
    lgraph_dl = removeLayers(lgraph_dl, layers(cls_out_idx).Name);
    dlnet = dlnetwork(lgraph_dl);

    % Also build a logits-only graph (softmax removed too), to test
    % whether softmax saturation is actually responsible for any small
    % gradient magnitudes below, rather than assuming it.
    sm_idx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.SoftmaxLayer'), layers), 1);
    lgraph_logits = layerGraph(net);
    lgraph_logits = removeLayers(lgraph_logits, {layers(cls_out_idx).Name, layers(sm_idx).Name});
    dlnet_logits = dlnetwork(lgraph_logits);

    for c = 1:n_classes
        % Restrict to samples whose TRUE label is c -- the same
        % population occlusion sensitivity uses for class c above, so
        % the two methods' correlation compares like with like rather
        % than a class-c-vs-all-samples mismatch.
        sel = find(roi_labels == class_names{c});
        Xc = dlarray(single(roi_data(:,:,:,sel)), 'SSCB');
        grad = extractdata(dlfeval(@gradient_for_class, dlnet, Xc, c));        % [H, W, 1, n_sel], w.r.t. softmax output
        grad_logits = extractdata(dlfeval(@gradient_for_class, dlnet_logits, Xc, c)); % w.r.t. pre-softmax logit
        % Signed mean (what gets plotted -- retains sign/direction, but
        % can be reduced by cross-sample cancellation) vs. mean of
        % per-sample absolute value (immune to cancellation, answers
        % "how large is the sensitivity typically", not "which
        % direction on average").
        grad_map(:, c) = mean(mean(grad, 2), 4);
        grad_absmean_map(:, c) = mean(mean(abs(grad), 2), 4);
        grad_logit_absmean_map(:, c) = mean(mean(abs(grad_logits), 2), 4);
    end
    grad_saliency_done = true;

    figure('Position', [100, 100, 900, 500]);
    yyaxis left;
    hold on;
    for c = 1:n_classes
        plot(wl, grad_map(:, c), 'LineWidth', 2, 'Color', class_colors(c,:), 'LineStyle', '-', 'DisplayName', class_names{c});
    end
    yline(0, 'k--', 'HandleVisibility', 'off');
    ax = gca; ax.YAxis(1).Color = 'k';
    xlabel('Wavelength (nm)'); ylabel('Mean input-gradient (signed, local sensitivity)');
    overlay_class_spectra(wl, class_avg_spectrum, n_classes, class_colors);
    title('Input-Gradient Saliency vs. Wavelength, with Average Spectra Overlaid (dotted)');
    yyaxis left;
    legend('Location', 'best');
    grid on;
    save_figure(gcf, fullfile(output_dir, 'gradient_saliency_vs_wavelength'));

    grad_occlusion_corr = zeros(n_classes, 1);
    cancellation_ratio = zeros(n_classes, 1);   % abs-mean / |signed-mean|: >1 means cross-sample sign cancellation is suppressing the signed average
    saturation_ratio = zeros(n_classes, 1);     % logit abs-mean / softmax-output abs-mean: >1 means the softmax layer itself is attenuating the gradient
    for c = 1:n_classes
        grad_occlusion_corr(c) = corr(grad_map(:,c), occlusion_map(:,c));
        cancellation_ratio(c) = max(grad_absmean_map(:,c)) / max(abs(grad_map(:,c)));
        saturation_ratio(c) = max(grad_logit_absmean_map(:,c)) / max(grad_absmean_map(:,c));
    end
    fprintf('  Correlation between gradient saliency and occlusion sensitivity, per class: %s\n', mat2str(grad_occlusion_corr', 3));
    fprintf('  Cross-sample cancellation ratio (abs-mean / |signed-mean|), per class: %s (>1 = signed average understates typical per-sample magnitude)\n', mat2str(cancellation_ratio', 3));
    fprintf('  Softmax-saturation ratio (logit-grad abs-mean / softmax-grad abs-mean), per class: %s (>1 = softmax layer is attenuating the gradient)\n', mat2str(saturation_ratio', 3));
catch err
    warning('cnn_weight_analysis:gradientSaliencyFailed', 'Skipping gradient saliency: %s', err.message);
    grad_saliency_done = false;
    grad_occlusion_corr = [];
    cancellation_ratio = [];
    saturation_ratio = [];
end

% --- Save numeric results + text summary ---
save_vars = {'filters', 'mean_act_per_wl', 'wl', 'avg_spectrum', 'corr_matrix', 'peak_vals', 'peak_idx', ...
    'class_mean_act', 'class_names', 'class_avg_spectrum', 'boundary_sensitive', 'interior_peak_wl', ...
    'separation_score', 'separation_metric_name', 'rank_corr', 'occlusion_map'};
if direct_path_done
    save_vars{end+1} = 'direct_path_weight';
end
if grad_saliency_done
    save_vars = [save_vars, {'grad_map', 'grad_absmean_map', 'grad_logit_absmean_map', ...
        'grad_occlusion_corr', 'cancellation_ratio', 'saturation_ratio'}];
end
save(fullfile(output_dir, 'cnn_analysis_results.mat'), save_vars{:});

fid = fopen(fullfile(output_dir, 'cnn_analysis_summary.txt'), 'w');
fprintf(fid, 'CNN Weight Analysis Summary (real trained network, real held-out data)\n');
fprintf(fid, '======================================================================\n\n');
fprintf(fid, 'Network file: %s\n', net_file);
fprintf(fid, 'Classes: %s\n', strjoin(class_names, ', '));
fprintf(fid, 'Held-out samples analyzed: %d\n\n', size(roi_data,4));
fprintf(fid, 'First conv layer ''%s'': %d filters, size %dx%dx%d\n\n', ...
    conv1_name, n_filters, size(filters,1), size(filters,2), size(filters,3));
fprintf(fid, 'Filters ranked by peak real activation (where each filter is most ENGAGED -- NOT the same as discriminative importance, see below):\n');
for i = 1:n_filters
    f = ranked(i);
    b = '';
    if boundary_sensitive(f)
        b = sprintf('  [PADDED BOUNDARY -- interior peak at %.1f nm]', interior_peak_wl(f));
    end
    fprintf(fid, '  Filter %d: peak activation %.4f at %.1f nm (corr with avg spectrum = %.3f)%s\n', ...
        f, peak_vals(f), wl(peak_idx(f)), corr_matrix(f), b);
end
fprintf(fid, '\nFilters ranked by class-separation (%s) -- which filters actually distinguish the classes:\n', separation_metric_name);
for i = 1:n_filters
    f = rank_by_separation(i);
    activation_rank = find(ranked == f);
    fprintf(fid, '  Filter %d: %s=%.4f (peak-activation rank was %d of %d)\n', ...
        f, separation_metric_name, separation_score(f), activation_rank, n_filters);
end
fprintf(fid, '\nCorrelation between activation magnitude and class-separation across filters: r=%.3f\n', rank_corr);
fprintf(fid, '(A low correlation means "most active" and "most discriminative" are different filters --\n');
fprintf(fid, ' discrimination likely emerges from a nonlinear combination of filters, not any single one.)\n');

if direct_path_done
    fprintf(fid, '\nDirect-path (Branch 2) LOCAL LINEAR APPROXIMATION of effective weight vs wavelength\n');
    fprintf(fid, '(fc_input->fc_final composition, treating the intermediate ReLU as identity -- this is an\n');
    fprintf(fid, 'approximation, NOT an exact decomposition; only the concat-channel index mapping used to\n');
    fprintf(fid, 'build it is exact, verified empirically with max match residual %.3g):\n', max(match_error));
    for c = 1:n_classes
        [~, mi] = max(abs(direct_path_weight(c,:)));
        fprintf(fid, '  %s: largest-magnitude weight %.4f at %.1f nm\n', class_names{c}, direct_path_weight(c,mi), wl(mi));
    end
else
    fprintf(fid, '\nDirect-path weight-vs-wavelength decomposition: skipped (see warning above).\n');
end

fprintf(fid, '\nOcclusion sensitivity vs wavelength (whole network; occluded rows replaced with the ROI''s own\n');
fprintf(fid, 'mean intensity, MaskValue=''auto'' -- verified empirically to NOT be zero; each class''s map is\n');
fprintf(fid, 'averaged only over held-out samples whose TRUE label is that class, not all held-out samples;\n');
fprintf(fid, 'positive means occluding that wavelength REDUCES confidence in the class, i.e. supporting\n');
fprintf(fid, 'evidence for it, in real examples of that class):\n');
for c = 1:n_classes
    [mx, mi] = max(occlusion_map(:,c));
    [mn, ni] = min(occlusion_map(:,c));
    fprintf(fid, '  %s: strongest positive evidence at %.1f nm (%.4f); strongest negative at %.1f nm (%.4f)\n', ...
        class_names{c}, wl(mi), mx, wl(ni), mn);
end

if grad_saliency_done
    fprintf(fid, '\nInput-gradient saliency vs wavelength (local sensitivity via backprop through the whole\n');
    fprintf(fid, 'network, including the max-pool layers; conditioned on true-label samples per class, same\n');
    fprintf(fid, 'as occlusion sensitivity above -- complementary to it, measuring a local derivative instead\n');
    fprintf(fid, 'of a large removal perturbation):\n');
    for c = 1:n_classes
        [mx, mi] = max(grad_map(:,c));
        [mn, ni] = min(grad_map(:,c));
        fprintf(fid, '  %s: strongest positive gradient at %.1f nm (%.4f); strongest negative at %.1f nm (%.4f)\n', ...
            class_names{c}, wl(mi), mx, wl(ni), mn);
    end
    fprintf(fid, '  Correlation with occlusion sensitivity, per class: %s\n', mat2str(grad_occlusion_corr', 3));
    fprintf(fid, '  Cross-sample cancellation ratio (abs-mean/|signed-mean|), per class: %s\n', mat2str(cancellation_ratio', 3));
    fprintf(fid, '    (>1 means the plotted signed average understates the typical per-sample |gradient| --\n');
    fprintf(fid, '     some of the "near-zero" appearance is samples'' gradients pointing in different\n');
    fprintf(fid, '     directions and cancelling in the average, not every sample having a tiny gradient.)\n');
    fprintf(fid, '  Softmax-saturation ratio (logit-grad abs-mean / softmax-grad abs-mean), per class: %s\n', mat2str(saturation_ratio', 3));
    fprintf(fid, '    (>1 means gradients computed through the pre-softmax logit are larger than through the\n');
    fprintf(fid, '     softmax output -- direct evidence the softmax layer itself is attenuating the signal,\n');
    fprintf(fid, '     i.e. genuine softmax saturation, rather than an assumption from magnitude alone.)\n');
else
    fprintf(fid, '\nInput-gradient saliency: skipped (see warning above).\n');
end
fclose(fid);

fprintf('\nCNN weight analysis complete. Results saved to %s/\n', output_dir);

end

function grad = gradient_for_class(dlnet, X, class_idx)
    Y = predict(dlnet, X);       % [n_classes x N], softmax probabilities
    Yc = Y(class_idx, :);
    Ysum = sum(Yc, 'all');       % batch-independent forward pass -> this still gives correct per-sample gradients
    grad = dlgradient(Ysum, X);
end

function overlay_class_spectra(wl, class_avg_spectrum, n_classes, class_colors)
    % Overlays each class's average real spectrum (dotted lines) on a
    % secondary right-hand y-axis of the CURRENT axes, explicitly using
    % the SAME class_colors palette as the left-axis curves. This is
    % necessary (not just convenient): under yyaxis, MATLAB does not
    % auto-cycle a distinct color per line the way plain axes do -- all
    % lines on one side default to a single shared color -- so colors
    % must be assigned explicitly on both sides to keep each class
    % identifiable and matched across the two axes.
    yyaxis right;
    hold on;
    % Normalized to a shared maximum across all classes (not per class, which
    % would erase real amplitude differences between dyes). See the units note
    % in occlusion_band_importance.m: the raw values are uncalibrated
    % background-subtracted detector counts, not photons, so the absolute
    % scale is not interpretable and this axis serves only as a positional
    % reference for where the emission band sits.
    scale = max(class_avg_spectrum(:));
    if ~isfinite(scale) || scale <= 0
        scale = 1;
    end
    for c = 1:n_classes
        plot(wl, class_avg_spectrum(:,c) / scale, 'LineStyle', ':', 'LineWidth', 1.5, ...
            'Color', class_colors(c,:), 'HandleVisibility', 'off');
    end
    ylim([0 1]);
    ylabel('Mean spectral intensity (normalized to shared maximum), dotted lines');
    ax = gca; ax.YAxis(2).Color = 'k';
end
