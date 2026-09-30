function cv_vs_singlesplit_figure(varargin)
% CV_VS_SINGLESPLIT_FIGURE
% Scatter of single-split against five-fold cross-validated accuracy for all
% 91 dye pairs, for the validation-robustness section of the reviewer
% response.
%
% The single-split baseline is results/original_single_split/pseudomix/matrices.mat,
% produced by the original analyze_bodipy_pairs.m. Note that its matrix is
% indexed by dye POSITION within target_bodipy = [1:7, 9:15], not by dye
% number -- dye 8 is unavailable -- so the two sources must be mapped
% through that vector rather than indexed directly.

p = inputParser;
% Resolve paths relative to this script's own location, so the script runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addParameter(p, 'base_dir', repo_root);
parse(p, varargin{:});
base_dir = p.Results.base_dir;

out_dir = fullfile(base_dir, 'results', 'cross_validation');
d = load(fullfile(base_dir, 'results', 'original_single_split', 'pseudomix', 'matrices.mat'));
A = d.cnn_accuracy;
target = [1:7, 9:15];

cvres = load(fullfile(base_dir, 'results', 'cross_validation', 'pairwise_cv_results.mat'));
R = cvres.all_results;
R = R(arrayfun(@(r) ~isnan(r.dye1) && ~r.skipped, R));

ss = []; cvv = []; sd = []; lbl = {};
for i = 1:numel(R)
    ia = find(target == R(i).dye1); ib = find(target == R(i).dye2);
    if isempty(ia) || isempty(ib), continue; end
    v = A(ia, ib);
    if ~isfinite(v) || v <= 0, v = A(ib, ia); end
    if ~isfinite(v) || v <= 0, continue; end
    ss(end+1) = v; %#ok<AGROW>
    cvv(end+1) = R(i).mean_accuracy; %#ok<AGROW>
    sd(end+1) = R(i).std_accuracy; %#ok<AGROW>
    lbl{end+1} = sprintf('%d-%d', R(i).dye1, R(i).dye2); %#ok<AGROW>
end

r = corr(ss(:), cvv(:));
figure('Position', [100 100 780 700]); hold on;
plot([0.45 0.95], [0.45 0.95], 'k--', 'LineWidth', 1.2, 'DisplayName', 'equality');
errorbar(ss, cvv, sd, 'o', 'MarkerSize', 5, 'LineStyle', 'none', ...
    'Color', [0.30 0.50 0.75], 'MarkerFaceColor', [0.30 0.50 0.75], ...
    'CapSize', 0, 'DisplayName', 'dye pair (bars = CV sd)');

% Label the three pairs discussed in the response text.
highlight = {'1-11', '1-13', '6-7'};
for k = 1:numel(highlight)
    j = find(strcmp(lbl, highlight{k}), 1);
    if isempty(j), continue; end
    plot(ss(j), cvv(j), 'o', 'MarkerSize', 9, 'LineWidth', 1.6, ...
        'Color', [0.85 0.33 0.10], 'HandleVisibility', 'off');
    text(ss(j)+0.008, cvv(j), lbl{j}, 'FontSize', 10, 'Color', [0.85 0.33 0.10]);
end

xlabel('Single-split accuracy (original analysis)');
ylabel('Five-fold cross-validated accuracy (mean \pm sd)');
title(sprintf('All 91 dye pairs: single-split vs. cross-validated accuracy (r = %.2f)', r));
axis([0.45 0.95 0.45 0.95]); axis square; grid on;
legend('Location', 'northwest');
save_figure(gcf, fullfile(out_dir, 'cv_vs_singlesplit'));
fprintf('r = %.4f, n = %d pairs; figure written to %s\n', r, numel(ss), ...
    fullfile(out_dir, 'cv_vs_singlesplit.png'));
end
