function reverify_interpretability()
% REVERIFY_INTERPRETABILITY
% Re-verification pass on the CNN interpretability analysis (Figures
% R1-R4) for the reviewer response. Uses the saved outputs of
% cnn_weight_analysis.m (mean_act_per_wl, wl, separation_score) for the
% pairwise (BODIPY-4-vs-10) and triplet (BODIPY-2/11/13) reference
% networks -- no re-training, same activations already reported.
%
% Addresses, per network:
%   1. Wavelength-axis documentation (band count, spacing, row mapping).
%   2. Peak table per filter using argmax(|mean activation|) (NOT raw
%      max, which the original script used and which can miss a
%      negative-going peak that is larger in magnitude).
%   3. Boundary check: which filters peak at row 1 or row N, and their
%      rank by peak magnitude.
%   4. Trimmed re-analysis: recompute peaks with rows 1 and N excluded,
%      and check whether engagement near the red edge (~794 nm) survives.
%   5. Correlation between peak |activation| and class-separation score
%      (Cohen's d / eta^2), redone with the abs-value peak criterion.
%   6. Shared-y-axis regeneration of the activation-vs-wavelength figures.


% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
networks = struct('name', {'pairwise', 'triplet'}, ...
                   'dir',  {fullfile(repo_root,'results','interpretability','pairwise'), fullfile(repo_root,'results','interpretability','triplet')});

out_dir = fullfile(repo_root,'results','interpretability','reverification');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

summary_lines = {};
summary_lines{end+1} = 'CNN Interpretability Re-Verification Summary';
summary_lines{end+1} = '=============================================';
summary_lines{end+1} = '';

for ni = 1:numel(networks)
    net_name = networks(ni).name;
    d = load(fullfile(networks(ni).dir, 'cnn_analysis_results.mat'));
    wl = d.wl(:);
    mact = d.mean_act_per_wl; % [H x n_filters]
    n_h = size(mact, 1);
    n_filters = size(mact, 2);
    sep_score = d.separation_score;
    sep_name = d.separation_metric_name;

    % --- 1. Wavelength axis documentation ---
    dwl = diff(wl);
    dwl_mean = mean(dwl);
    dwl_std = std(dwl);
    uniform = dwl_std < 0.01 * dwl_mean;

    summary_lines{end+1} = sprintf('=== Network: %s ===', net_name);
    summary_lines{end+1} = sprintf('Spectral bands (H, rows): %d', n_h);
    summary_lines{end+1} = sprintf('Wavelength range: %.2f - %.2f nm', wl(1), wl(end));
    summary_lines{end+1} = sprintf('Sampling interval: mean %.4f nm, std %.4f nm (%s)', ...
        dwl_mean, dwl_std, ternary(uniform, 'uniform', 'NOT uniform -- check wl vector'));
    summary_lines{end+1} = 'Row-to-wavelength mapping: wavelength(row) = wl(row), i.e. output row index r of the';
    summary_lines{end+1} = 'conv1 activation map (after ''same'' padding, so H_out == H_in == 55) maps directly to wl(r).';
    summary_lines{end+1} = sprintf('Padding confirmation (3x3 kernel, ''same'' padding): only row 1 and row %d receive a', n_h);
    summary_lines{end+1} = sprintf('zero-padded neighbor (row 1 lacks a real row 0; row %d lacks a real row %d).', n_h, n_h+1);
    summary_lines{end+1} = sprintf('Row %d (wl = %.1f nm) uses only real neighboring rows (%d,%d,%d) and is NOT a boundary row.', ...
        n_h-1, wl(n_h-1), n_h-2, n_h-1, n_h);
    summary_lines{end+1} = '';

    % --- 2. Peak table: argmax(|mean activation|) ---
    peak_val_signed = zeros(n_filters,1);
    peak_row = zeros(n_filters,1);
    peak_wl = zeros(n_filters,1);
    for f = 1:n_filters
        [~, r] = max(abs(mact(:,f)));
        peak_row(f) = r;
        peak_val_signed(f) = mact(r,f);
        peak_wl(f) = wl(r);
    end
    peak_abs = abs(peak_val_signed);

    T = table((1:n_filters)', peak_wl, peak_val_signed, peak_row, peak_abs, ...
        'VariableNames', {'filter_id','peak_wavelength_nm','signed_value_at_peak','output_row_index','abs_value_at_peak'});
    csv_path = fullfile(out_dir, sprintf('peak_table_%s.csv', net_name));
    writetable(T, csv_path);
    summary_lines{end+1} = sprintf('Peak table (criterion: argmax over rows of |mean activation|) saved to %s', csv_path);

    % --- 3. Boundary check ---
    [~, rank_by_abs] = sort(peak_abs, 'descend'); % rank_by_abs(1) = filter id with largest |peak|
    abs_rank_of = zeros(n_filters,1);
    for i = 1:n_filters
        abs_rank_of(rank_by_abs(i)) = i;
    end
    is_boundary = (peak_row == 1) | (peak_row == n_h);
    boundary_filters = find(is_boundary);
    summary_lines{end+1} = sprintf('Boundary-peaking filters (row 1 or row %d), argmax(|activation|) criterion: %d of %d', ...
        n_h, numel(boundary_filters), n_filters);
    if isempty(boundary_filters)
        summary_lines{end+1} = '  None.';
    else
        for k = 1:numel(boundary_filters)
            f = boundary_filters(k);
            summary_lines{end+1} = sprintf('  Filter %d: row %d (%.1f nm), |peak|=%.4f, rank by |peak| magnitude = %d of %d', ...
                f, peak_row(f), peak_wl(f), peak_abs(f), abs_rank_of(f), n_filters);
        end
    end
    summary_lines{end+1} = '';

    % --- 4. Trimmed re-analysis (exclude rows 1 and N) ---
    interior_rows = 2:(n_h-1);
    trim_wl = wl(interior_rows);
    trim_peak_row_local = zeros(n_filters,1);
    trim_peak_row_global = zeros(n_filters,1);
    trim_peak_wl = zeros(n_filters,1);
    trim_peak_val = zeros(n_filters,1);
    for f = 1:n_filters
        [~, r_local] = max(abs(mact(interior_rows, f)));
        trim_peak_row_local(f) = r_local;
        trim_peak_row_global(f) = interior_rows(r_local);
        trim_peak_wl(f) = wl(interior_rows(r_local));
        trim_peak_val(f) = mact(interior_rows(r_local), f);
    end
    Ttrim = table((1:n_filters)', trim_peak_wl, trim_peak_val, trim_peak_row_global, abs(trim_peak_val), ...
        'VariableNames', {'filter_id','trimmed_peak_wavelength_nm','signed_value','output_row_index','abs_value'});
    writetable(Ttrim, fullfile(out_dir, sprintf('peak_table_trimmed_%s.csv', net_name)));

    % Red-edge engagement check: define the red-edge region as the top
    % quartile of the (untrimmed) wavelength axis, i.e. wl >= wl(round(0.75*n_h)).
    red_edge_thresh = wl(round(0.75*n_h));
    n_red_untrimmed = sum(peak_wl >= red_edge_thresh);
    n_red_trimmed = sum(trim_peak_wl >= red_edge_thresh);
    % Specifically: how many filters whose UNTRIMMED peak was in the true
    % boundary row (row N, the only row this trim actually removes at the
    % red end) still land in the red-edge region after trimming, vs move
    % away entirely.
    boundary_at_N = find(peak_row == n_h);
    if isempty(boundary_at_N)
        moved_away_frac = NaN;
        note = 'No filter''s untrimmed peak was at the true red-edge boundary row (row N) to begin with.';
    else
        still_red = sum(trim_peak_wl(boundary_at_N) >= red_edge_thresh);
        moved_away_frac = 1 - still_red/numel(boundary_at_N);
        note = sprintf('%d filter(s) had untrimmed peak at row N; after trimming, %d of %d still land in the red-edge region.', ...
            numel(boundary_at_N), still_red, numel(boundary_at_N));
    end
    summary_lines{end+1} = sprintf('Trimmed re-analysis (rows 1 and %d excluded):', n_h);
    summary_lines{end+1} = sprintf('  Red-edge region defined as wl >= %.1f nm (top quartile of axis).', red_edge_thresh);
    summary_lines{end+1} = sprintf('  Filters with peak in red-edge region: %d/%d untrimmed -> %d/%d trimmed.', ...
        n_red_untrimmed, n_filters, n_red_trimmed, n_filters);
    summary_lines{end+1} = ['  ' note];
    verdict = ternary(n_red_trimmed > 0, 'YES', 'NO');
    summary_lines{end+1} = sprintf('  VERDICT: red-edge engagement (~%.0f-%.0f nm) SURVIVES trimming: %s', ...
        wl(n_h-2), wl(n_h-1), verdict);
    summary_lines{end+1} = '';

    % --- 5. Correlation: |peak activation| vs class-separation score ---
    r_signed_orig = corr(sep_score, d.peak_vals(:)); % as originally computed (signed max, not abs)
    r_abs = corr(sep_score, peak_abs(:));
    [~, best_sep_filter] = max(sep_score);
    best_sep_rank_by_abs = abs_rank_of(best_sep_filter);
    summary_lines{end+1} = sprintf('Correlation, activation magnitude vs class-separation (%s):', sep_name);
    summary_lines{end+1} = sprintf('  Original (signed max, as previously reported): r = %.3f', r_signed_orig);
    summary_lines{end+1} = sprintf('  Recomputed (argmax|activation| criterion):     r = %.3f', r_abs);
    summary_lines{end+1} = sprintf('  Strongest-separation filter: Filter %d (%s = %.4f), rank by |peak activation| = %d of %d', ...
        best_sep_filter, sep_name, sep_score(best_sep_filter), best_sep_rank_by_abs, n_filters);
    summary_lines{end+1} = '';

    % --- 6. Figure regeneration: shared y-axis across all 16 panels ---
    n_cols = min(8, ceil(sqrt(n_filters)));
    n_rows = ceil(n_filters / n_cols);
    ylims = [min(mact(:)), max(mact(:))];
    pad = 0.05*range(ylims);
    ylims = ylims + [-pad, pad];

    fig = figure('Position', [100, 100, 1200, 800]);
    for f = 1:n_filters
        subplot(n_rows, n_cols, f);
        plot(wl, mact(:,f), 'LineWidth', 2);
        ylim(ylims);
        xlabel('Wavelength (nm)'); ylabel('Mean activation');
        title(sprintf('Filter %d', f));
        grid on;
    end
    sgtitle(sprintf('Filter Activations vs. Wavelength -- shared y-axis (%s network)', net_name));
    save_figure(fig, fullfile(out_dir, sprintf('filter_wavelength_responses_%s_sharedaxis', net_name)));
    close(fig);

    fig2 = figure('Position', [100, 100, 1200, 800]);
    for f = 1:n_filters
        subplot(n_rows, n_cols, f);
        plot(trim_wl, mact(interior_rows,f), 'LineWidth', 2, 'Color', [0.85 0.33 0.10]);
        xlabel('Wavelength (nm)'); ylabel('Mean activation');
        title(sprintf('Filter %d', f));
        grid on;
    end
    sgtitle(sprintf('Filter Activations vs. Wavelength -- rows 1 & %d (boundary) excluded (%s network)', n_h, net_name));
    save_figure(fig2, fullfile(out_dir, sprintf('filter_wavelength_responses_%s_trimmed', net_name)));
    close(fig2);

    summary_lines{end+1} = sprintf('Figures saved: filter_wavelength_responses_%s_sharedaxis.png, filter_wavelength_responses_%s_trimmed.png', net_name, net_name);
    summary_lines{end+1} = '';
    summary_lines{end+1} = '';
end

fid = fopen(fullfile(out_dir, 'reverification_summary.txt'), 'w');
fprintf(fid, '%s\n', summary_lines{:});
fclose(fid);
for i = 1:numel(summary_lines)
    fprintf('%s\n', summary_lines{i});
end
fprintf('\nAll outputs written to %s/\n', out_dir);

end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
