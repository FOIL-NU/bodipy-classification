function save_figure(fig, basepath, style)
% SAVE_FIGURE
% Exports a figure as BOTH a raster PNG (for the response documents, which
% embed images) and a vector EPS (for editing in Illustrator), using the lab
% formatfig utility so that fonts and line widths match house style.
%
%   save_figure(gcf, fullfile(out_dir, 'my_figure'))
%   save_figure(gcf, fullfile(out_dir, 'my_figure'), 'publication')
%
% basepath is given WITHOUT an extension; '.png' and '.eps' are appended.
%
% WHY THE RENDERER IS SET HERE RATHER THAN LEFT TO formatfig
% There are two formatfig implementations in play. The internal lab version
% takes a fourth 'painters' argument that sets the Painters renderer; the
% stub shipped here in src/original_single_split/ takes only three arguments
% and would error if given a fourth. Whichever one happens to be first on the path,
% this function calls formatfig with three arguments only and sets the
% renderer itself. That matters for more than compatibility: without the
% Painters renderer MATLAB may rasterize the figure contents inside the EPS,
% which produces a file that opens in Illustrator but contains a bitmap
% rather than editable paths -- exactly what an EPS export is wanted to
% avoid here.
%
% formatfig is applied BEFORE the PNG is written so that both files carry
% the same fonts and line widths; otherwise the raster and vector versions
% of the same figure would not match.
%
% If formatfig is not on the path, the export still succeeds through a
% direct print with -painters, and a warning says the house style was not
% applied.

if nargin < 1 || isempty(fig), fig = gcf; end
if nargin < 3 || isempty(style), style = 'presentation'; end

[outdir, ~, ext] = fileparts(basepath);
if ~isempty(ext) && any(strcmpi(ext, {'.png', '.eps', '.svg'}))
    % Tolerate a path passed with an extension by stripping it, so callers
    % cannot accidentally produce 'name.png.eps'.
    basepath = fullfile(outdir, erase(basepath, ext));
    basepath = regexprep(basepath, [regexptranslate('escape', ext) '$'], '');
end
if ~isempty(outdir) && ~exist(outdir, 'dir')
    mkdir(outdir);
end

eps_path = [basepath '.eps'];
png_path = [basepath '.png'];

% Vector output: Painters emits paths rather than a rasterized image.
set(fig, 'Renderer', 'painters');

if exist('formatfig', 'file') == 2
    try
        formatfig(fig, style, eps_path);
    catch ME
        warning('save_figure:formatfig', ...
            'formatfig failed (%s); falling back to a direct EPS export.', ME.message);
        print(fig, eps_path, '-depsc', '-painters');
    end
else
    warning('save_figure:noFormatfig', ...
        'formatfig not found on the path; exporting EPS without house-style formatting.');
    print(fig, eps_path, '-depsc', '-painters');
end

print(fig, png_path, '-dpng', '-r300');
end
