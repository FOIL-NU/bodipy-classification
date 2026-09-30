function reverify_interpretability_round2()
% Round 2 re-verification, addressing 5 follow-up requests:
%  1. Trimmed red-edge response magnitude, reported as numbers (not just counts).
%  2. Random-initialization control: same forward pass, untrained conv1 filters.
%  3. Activation normalized by per-band input amplitude/variance.
%  4. Onset-cluster count with the boundary-peaking filters excluded entirely.
%  5. Both peak locations per filter (largest +excursion and largest -excursion).


% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
nets = struct('name', {'pairwise','triplet'}, ...
              'ref_file', {fullfile(repo_root,'results','cross_validation','pairwise_reference_net.mat'), fullfile(repo_root,'results','cross_validation','triplet_reference_net.mat')}, ...
              'analysis_dir', {fullfile(repo_root,'results','interpretability','pairwise'), fullfile(repo_root,'results','interpretability','triplet')});

out_dir = fullfile(repo_root,'results','interpretability','reverification');
if ~exist(out_dir,'dir'); mkdir(out_dir); end

n_random_trials = 30;
rng(12345); % fixed seed for reproducibility

lines = {};
lines{end+1} = 'CNN Interpretability Re-Verification -- Round 2 Summary';
lines{end+1} = '==========================================================';
lines{end+1} = '';

for ni = 1:numel(nets)
    net_name = nets(ni).name;
    ref = load(nets(ni).ref_file);
    an  = load(fullfile(nets(ni).analysis_dir, 'cnn_analysis_results.mat'));
    wl = ref.wl(:);
    n_h = numel(wl);
    roi_data = double(ref.roi_data); % [H W 1 N]
    n_filters = size(an.mean_act_per_wl, 2);
    mact_trained = an.mean_act_per_wl; % [H x n_filters], TRAINED, real activations (as before)

    conv1 = ref.net.Layers(arrayfun(@(x) isa(x,'nnet.cnn.layer.Convolution2DLayer'), ref.net.Layers));
    conv1 = conv1(1);
    input_mean = ref.net.Layers(1).Mean; % zero-centering applied by ImageInputLayer

    lines{end+1} = sprintf('=== Network: %s ===', net_name);

    % ---------- Item 5: both peak locations per filter ----------
    pos_wl = zeros(n_filters,1); pos_val = zeros(n_filters,1); pos_row = zeros(n_filters,1);
    neg_wl = zeros(n_filters,1); neg_val = zeros(n_filters,1); neg_row = zeros(n_filters,1);
    for f = 1:n_filters
        [pv, pr] = max(mact_trained(:,f));
        [nv, nr] = min(mact_trained(:,f));
        pos_wl(f) = wl(pr); pos_val(f) = pv; pos_row(f) = pr;
        neg_wl(f) = wl(nr); neg_val(f) = nv; neg_row(f) = nr;
    end
    T5 = table((1:n_filters)', pos_wl, pos_val, pos_row, neg_wl, neg_val, neg_row, ...
        'VariableNames', {'filter_id','pos_peak_wl_nm','pos_peak_val','pos_peak_row', ...
                           'neg_peak_wl_nm','neg_peak_val','neg_peak_row'});
    writetable(T5, fullfile(out_dir, sprintf('dual_peak_table_%s.csv', net_name)));
    n_both_in_two_regions = sum((pos_wl >= 599 & pos_wl <= 625) & (neg_wl >= 780 | ...
        (neg_wl >= 599 & neg_wl <= 625))) ; % informational; see text below for the real tally
    onset_lo = 599; onset_hi = 625; red_lo = 780;
    pos_in_onset = pos_wl >= onset_lo & pos_wl <= onset_hi;
    neg_in_onset = neg_wl >= onset_lo & neg_wl <= onset_hi;
    pos_in_red = pos_wl >= red_lo;
    neg_in_red = neg_wl >= red_lo;
    n_onset_either = sum(pos_in_onset | neg_in_onset);
    n_red_either = sum(pos_in_red | neg_in_red);
    n_both_regions = sum((pos_in_onset & neg_in_red) | (pos_in_red & neg_in_onset));
    lines{end+1} = sprintf('Dual-peak table saved: dual_peak_table_%s.csv (pos + neg excursion per filter)', net_name);
    lines{end+1} = sprintf('  Filters with EITHER excursion in onset band (599-625 nm): %d of %d', n_onset_either, n_filters);
    lines{end+1} = sprintf('  Filters with EITHER excursion in red-edge band (>=780 nm): %d of %d', n_red_either, n_filters);
    lines{end+1} = sprintf('  Filters with ONE excursion in onset AND the OTHER in red-edge (genuine two-region support): %d of %d', n_both_regions, n_filters);

    % ---------- Item 4: onset-cluster count excluding boundary-peaking filters ----------
    peak_abs = zeros(n_filters,1); peak_row_abs = zeros(n_filters,1); peak_wl_abs = zeros(n_filters,1);
    for f = 1:n_filters
        [~, r] = max(abs(mact_trained(:,f)));
        peak_row_abs(f) = r; peak_wl_abs(f) = wl(r); peak_abs(f) = mact_trained(r,f);
    end
    is_boundary = (peak_row_abs == 1) | (peak_row_abs == n_h);
    non_boundary = find(~is_boundary);
    in_onset_nb = sum(peak_wl_abs(non_boundary) >= onset_lo & peak_wl_abs(non_boundary) <= onset_hi);
    lines{end+1} = sprintf('Onset-cluster count with boundary-peaking filters removed: %d of %d remaining filters peak in 599-625 nm (excluded: %s)', ...
        in_onset_nb, numel(non_boundary), mat2str(find(is_boundary)'));

    % ---------- Item 1: trimmed red-edge magnitude, as numbers ----------
    interior = 2:(n_h-1);
    red_edge_rows = interior(wl(interior) >= 770); % interior rows only, near red edge, boundary row 55 excluded
    onset_rows = interior(wl(interior) >= onset_lo & wl(interior) <= onset_hi);
    red_mag = max(abs(mact_trained(red_edge_rows, :)), [], 1); % 1 x n_filters
    onset_mag = max(abs(mact_trained(onset_rows, :)), [], 1);
    ratio = red_mag ./ onset_mag;
    lines{end+1} = sprintf('Trimmed (rows 1 & %d excluded) magnitude comparison, red-edge (>=770 nm, interior only) vs onset (599-625 nm):', n_h);
    lines{end+1} = sprintf('  Median |red-edge| max activation across filters: %.2f', median(red_mag));
    lines{end+1} = sprintf('  Median |onset| max activation across filters:    %.2f', median(onset_mag));
    lines{end+1} = sprintf('  Median ratio (red-edge / onset), per filter: %.3f  (i.e. red-edge response is typically %.0f%% of onset response)', ...
        median(ratio), 100*median(ratio));
    lines{end+1} = sprintf('  Range of per-filter ratio: %.3f - %.3f', min(ratio), max(ratio));

    % ---------- Item 3: activation normalized by per-band input amplitude/variance ----------
    % Per-band input variability, computed on the SAME zero-centered input the network sees.
    roi_centered = roi_data - input_mean;
    band_std = squeeze(std(reshape(roi_centered, n_h, []), 0, 2)); % [n_h x 1], std over (W,N) per row
    norm_mact = mact_trained ./ band_std; % normalize each row's activation by that band's own input variability
    norm_peak_wl = zeros(n_filters,1);
    for f = 1:n_filters
        [~, r] = max(abs(norm_mact(:,f)));
        norm_peak_wl(f) = wl(r);
    end
    n_norm_onset = sum(norm_peak_wl >= onset_lo & norm_peak_wl <= onset_hi);
    n_norm_red = sum(norm_peak_wl >= red_lo);
    lines{end+1} = sprintf('Variance-normalized activation (mean_act / per-band input std): peak location shifts to onset band for %d/%d filters, red-edge for %d/%d filters', ...
        n_norm_onset, n_filters, n_norm_red, n_filters);
    lines{end+1} = sprintf('  (Band std range across the 55 bands: %.2f - %.2f; onset-band mean std %.2f vs red-edge-band mean std %.2f)', ...
        min(band_std), max(band_std), mean(band_std(wl>=onset_lo & wl<=onset_hi)), mean(band_std(wl>=770)));

    % ---------- Item 2: random-initialization control ----------
    fh = conv1.FilterSize(1); fw = conv1.FilterSize(2);
    n_ch = conv1.NumChannels;
    fan_in = fh*fw*n_ch;
    X = dlarray(single(roi_centered), 'SSCB');
    rand_peak_wl_all = zeros(n_random_trials*n_filters, 1);
    k = 0;
    for trial = 1:n_random_trials
        Wt = dlarray(single(randn(fh, fw, n_ch, n_filters) * sqrt(2/fan_in))); % He init
        Bt = dlarray(zeros(1,1,n_filters,'single'));
        act_rand = extractdata(dlconv(X, Wt, Bt, 'Padding', 'same'));  % [H W n_filters N]
        mact_rand = squeeze(mean(mean(act_rand, 2), 4)); % [H x n_filters]
        for f = 1:n_filters
            [~, r] = max(abs(mact_rand(:,f)));
            k = k+1;
            rand_peak_wl_all(k) = wl(r);
        end
    end
    n_rand_onset = sum(rand_peak_wl_all >= onset_lo & rand_peak_wl_all <= onset_hi);
    n_rand_red = sum(rand_peak_wl_all >= red_lo);
    n_rand_total = numel(rand_peak_wl_all);
    lines{end+1} = sprintf('RANDOM-INIT CONTROL (%d trials x %d untrained filters = %d instances, He-init, same forward pass, same real ROI data):', ...
        n_random_trials, n_filters, n_rand_total);
    lines{end+1} = sprintf('  Untrained-filter peaks landing in onset band (599-625 nm): %d of %d (%.1f%%)', ...
        n_rand_onset, n_rand_total, 100*n_rand_onset/n_rand_total);
    lines{end+1} = sprintf('  Untrained-filter peaks landing in red-edge band (>=780 nm): %d of %d (%.1f%%)', ...
        n_rand_red, n_rand_total, 100*n_rand_red/n_rand_total);
    lines{end+1} = sprintf('  For comparison, TRAINED filters landing in onset band: %d of %d (%.1f%%)', ...
        sum(peak_wl_abs>=onset_lo & peak_wl_abs<=onset_hi), n_filters, 100*sum(peak_wl_abs>=onset_lo & peak_wl_abs<=onset_hi)/n_filters);

    lines{end+1} = '';
end

fid = fopen(fullfile(out_dir, 'reverification_round2_summary.txt'), 'w');
fprintf(fid, '%s\n', lines{:});
fclose(fid);
for i = 1:numel(lines)
    fprintf('%s\n', lines{i});
end
fprintf('\nAll round-2 outputs written to %s/\n', out_dir);
end
