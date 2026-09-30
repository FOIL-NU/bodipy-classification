% Test script for spectral feature extraction
% Uses the same synthetic spectra as earlier drafts of this analysis so
% the shoulder-detection fix can be directly compared: samples 3 and 10
% are deliberately bimodal/multi-peak and should now show non-zero
% shoulder_intensity / shoulder_position.
clear; clc; close all;

addpath(pwd);
rng(42, 'twister');

fprintf('=== Spectral Feature Extraction Demo ===\n\n');

wl = 600:5:800; % Wavelengths from 600 to 800 nm (41 points)
n_samples = 10;

spectral_profiles = zeros(length(wl), n_samples);

spectral_profiles(:,1) = exp(-((wl-650).^2)/(2*20^2));                                 % sharp peak
spectral_profiles(:,2) = exp(-((wl-700).^2)/(2*40^2));                                 % broad peak
spectral_profiles(:,3) = 0.7*exp(-((wl-650).^2)/(2*20^2)) + 0.5*exp(-((wl-720).^2)/(2*25^2)); % bimodal
spectral_profiles(:,4) = exp(-((wl-680).^2)/(2*25^2)) .* (1 + 0.3*(wl-680)/100);        % asymmetric
spectral_profiles(:,5) = 0.1 + 0.5*exp(-((wl-700).^2)/(2*30^2));                        % flat + small peak
spectral_profiles(:,6) = exp(-((wl-620).^2)/(2*25^2));                                 % left side
spectral_profiles(:,7) = exp(-((wl-750).^2)/(2*25^2));                                 % right side
spectral_profiles(:,8) = exp(-((wl-700).^2)/(2*15^2));                                 % narrow
spectral_profiles(:,9) = exp(-((wl-650).^2)/(2*60^2));                                 % very broad
spectral_profiles(:,10) = 0.5*exp(-((wl-640).^2)/(2*10^2)) + 0.3*exp(-((wl-680).^2)/(2*10^2)) + 0.2*exp(-((wl-720).^2)/(2*10^2)); % multi-peak (narrower, resolvable components)

[features, feature_names] = extract_spectral_features(spectral_profiles, wl);

fprintf('Extracted %d features for %d samples:\n\n', size(features,2), size(features,1));
fprintf('%20s', 'Sample');
for f = 1:size(features,2)
    fprintf('  %15s', feature_names{f});
end
fprintf('\n');
for s = 1:size(features,1)
    fprintf('%20d', s);
    for f = 1:size(features,2)
        fprintf('  %15.2f', features(s,f));
    end
    fprintf('\n');
end

fprintf('\nSanity check -- deliberately bimodal/multi-peak samples (3, 10) should have non-zero shoulder_intensity:\n');
fprintf('  Sample 3 (bimodal 650/720):  shoulder_intensity = %.3f, shoulder_position = %.1f nm\n', features(3,5), features(3,6));
fprintf('  Sample 10 (multi-peak):      shoulder_intensity = %.3f, shoulder_position = %.1f nm\n', features(10,5), features(10,6));

% Resolve paths relative to this script's location
% (repo_root/src/interpretability/<this file>), so it runs correctly
% regardless of MATLAB's current working directory.
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(script_dir));
addpath(script_dir);
output_dir = fullfile(repo_root, 'results', 'interpretability', 'demo');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

f1 = figure('Position', [100, 100, 1000, 600]);
colors = lines(n_samples);
for s = 1:n_samples
    plot(wl, spectral_profiles(:,s), 'LineWidth', 2, 'Color', colors(s,:));
    hold on;
end
xlabel('Wavelength (nm)');
ylabel('Intensity');
title('Synthetic Spectral Profiles');
grid on;
legend(arrayfun(@(x) sprintf('Sample %d', x), 1:n_samples, 'UniformOutput', false));
save_figure(f1, fullfile(output_dir, 'synthetic_spectra'));

f3 = figure('Position', [100, 100, 900, 700]);
features_norm = zscore(features);
corr_matrix = corr(features_norm);
imagesc(corr_matrix);
colorbar;
xticks(1:size(features,2)); yticks(1:size(features,2));
xticklabels(feature_names); yticklabels(feature_names);
xtickangle(45);
title('Feature Correlation Matrix');
save_figure(f3, fullfile(output_dir, 'feature_correlation'));

save(fullfile(output_dir, 'synthetic_features.mat'), 'features', 'feature_names', 'wl', 'spectral_profiles');

fprintf('\n=== Feature Extraction Demo Complete ===\n');
