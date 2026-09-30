function formatfig(fig, style, filepath) %#ok<INUSL>
% FORMATFIG - Minimal figure-export helper.
%
% The original scripts in this folder call an internal lab utility of
% this name that formats a figure for a given STYLE (e.g.
% 'presentation') and exports it to FILEPATH. That utility is not part
% of this public release. This is a simplified stand-in: it creates the
% output directory if needed and exports the figure in the format
% implied by FILEPATH's extension (.png or .eps; anything else falls
% back to saveas).
%
% Substituting this function does not change any computed
% classification metric (accuracy/precision/recall/F1) -- every call
% to it in these scripts happens after the relevant metric has already
% been computed and, where applicable, saved to a .mat file. It only
% affects the cosmetic appearance/export of supplementary figures, so
% figures regenerated with this stub may differ visually from the
% originals (font sizes, colors, etc.) without affecting reproducibility
% of the reported numbers.
%
% Inputs:
%   fig      - figure handle
%   style    - ignored (kept for call-signature compatibility)
%   filepath - output file path; parent directory is created if needed

outdir = fileparts(filepath);
if ~isempty(outdir) && ~exist(outdir, 'dir')
    mkdir(outdir);
end

[~, ~, ext] = fileparts(filepath);
switch lower(ext)
    case '.png'
        print(fig, filepath, '-dpng', '-r300');
    case '.eps'
        print(fig, filepath, '-depsc', '-r300');
    otherwise
        saveas(fig, filepath);
end

end
