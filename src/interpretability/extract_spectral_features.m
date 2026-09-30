function [features, feature_names] = extract_spectral_features(spectral_profiles, wl, varargin)
% EXTRACT_SPECTRAL_FEATURES
% Extract key spectral features from spectral profiles
%
% Inputs:
%   spectral_profiles - 2D matrix [wavelengths x samples] or 3D/4D with
%                        spatial dimensions to be averaged out
%   wl - wavelength vector
%   varargin - optional: 'min_peak_height' (default 0.05, fraction of max),
%              'min_peak_prominence' (default 0.05, fraction of max)
%
% Outputs:
%   features - matrix [n_samples x n_features] of extracted features
%   feature_names - cell array of feature names
%
% Features extracted:
%   1. Peak position (wavelength of maximum)
%   2. Peak intensity (maximum value)
%   3. Spectral centroid (weighted average wavelength)
%   4. FWHM (Full Width at Half Maximum)
%   5. Shoulder intensity (second peak intensity)
%   6. Shoulder position (wavelength of second peak)
%   7. Asymmetry (left-right balance)
%   8. Kurtosis (peakedness)
%   9. Band width at 90% of max
%   10. Red edge integral (integral of red side)
%
% NOTE ON SHOULDER DETECTION: earlier versions of this function passed
% findpeaks' 'Threshold' name-value pair a value scaled to the spectrum's
% max intensity. 'Threshold' is NOT a relative-height cutoff -- it is the
% minimum absolute difference between a peak and its immediate
% neighboring samples. For smoothly-sampled spectra that difference is
% always tiny, so a threshold of ~0.5 silently suppressed every
% secondary peak, including on deliberately bimodal test spectra. This
% version uses 'MinPeakProminence' instead, which is the correct
% relative-height concept for shoulder/secondary-peak detection.

p = inputParser;
addParameter(p, 'min_peak_height', 0.05);
addParameter(p, 'min_peak_prominence', 0.05);
parse(p, varargin{:});
min_peak_height = p.Results.min_peak_height;
min_peak_prominence = p.Results.min_peak_prominence;

% Handle different input dimensions -- collapse any spatial dimensions,
% keep [wavelengths x samples]
if ndims(spectral_profiles) == 4
    spectral_profiles = squeeze(mean(mean(spectral_profiles, 1), 2));
elseif ndims(spectral_profiles) == 3
    if size(spectral_profiles, 2) == length(wl)
        spectral_profiles = squeeze(mean(spectral_profiles, 1));
    else
        spectral_profiles = squeeze(mean(spectral_profiles, 2));
    end
end

n_samples = size(spectral_profiles, 2);

features = zeros(n_samples, 10);
feature_names = {
    'peak_position', ...
    'peak_intensity', ...
    'spectral_centroid', ...
    'fwhm', ...
    'shoulder_intensity', ...
    'shoulder_position', ...
    'asymmetry', ...
    'kurtosis', ...
    'band_width_90', ...
    'red_edge_integral'
};

% Normalize each spectrum to [0,1] for consistent feature extraction
spectral_profiles_norm = spectral_profiles ./ max(spectral_profiles, [], 1);

wl_col = wl(:);

for s = 1:n_samples
    profile = spectral_profiles(:, s);
    profile_norm = spectral_profiles_norm(:, s);

    % 1. Peak position (wavelength of maximum)
    [max_val, max_idx] = max(profile);
    features(s, 1) = wl_col(max_idx);

    % 2. Peak intensity (maximum value)
    features(s, 2) = max_val;

    % 3. Spectral centroid (weighted average wavelength)
    if sum(profile) > 0
        features(s, 3) = sum(wl_col .* profile) / sum(profile);
    else
        features(s, 3) = wl_col(round(length(wl_col)/2));
    end

    % 4. FWHM (Full Width at Half Maximum)
    half_max = max_val / 2;
    above_half = profile >= half_max;
    if sum(above_half) >= 2
        first_idx = find(above_half, 1, 'first');
        last_idx = find(above_half, 1, 'last');
        features(s, 4) = wl_col(last_idx) - wl_col(first_idx);
    else
        features(s, 4) = 0;
    end

    % 5-6. Shoulder detection (second-highest local maximum), using
    % MinPeakProminence -- a relative-height concept -- rather than the
    % previous (broken) absolute 'Threshold'.
    [pks, locs] = findpeaks(profile_norm, 'MinPeakHeight', min_peak_height, ...
        'MinPeakDistance', 5, 'MinPeakProminence', min_peak_prominence);

    if length(pks) >= 2
        [sorted_pks, sorted_idx] = sort(pks, 'descend');
        features(s, 5) = sorted_pks(2);
        features(s, 6) = wl_col(locs(sorted_idx(2)));
    else
        features(s, 5) = 0;
        features(s, 6) = 0;
    end

    % 7. Asymmetry (difference between left and right half-max widths)
    left_idx = find(profile_norm(1:max_idx) >= 0.5 * profile_norm(max_idx), 1, 'first');
    right_idx_rel = find(profile_norm(max_idx:end) >= 0.5 * profile_norm(max_idx), 1, 'last');
    if ~isempty(left_idx) && ~isempty(right_idx_rel)
        left_width = max_idx - left_idx;
        right_width = right_idx_rel - 1;
        if (left_width + right_width) > 0
            features(s, 7) = (right_width - left_width) / (right_width + left_width);
        else
            features(s, 7) = 0;
        end
    else
        features(s, 7) = 0;
    end

    % 8. Kurtosis of the profile
    features(s, 8) = kurtosis(profile);

    % 9. Band width at 90% of max
    threshold_90 = 0.9 * max_val;
    above_90 = profile >= threshold_90;
    if sum(above_90) >= 2
        first_90 = find(above_90, 1, 'first');
        last_90 = find(above_90, 1, 'last');
        features(s, 9) = wl_col(last_90) - wl_col(first_90);
    else
        features(s, 9) = 0;
    end

    % 10. Red edge integral (integral of the red/long-wavelength side)
    red_profile = profile(max_idx:end);
    red_wl = wl_col(max_idx:end);
    if length(red_profile) >= 2
        features(s, 10) = trapz(red_wl, red_profile);
    else
        features(s, 10) = 0;
    end
end

end
