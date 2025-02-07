function [] = fig03(xi,x2,psi)
% addpath '/Users/rss367/Documents/MATLAB/matlab-dl-transformer/rs/other models';
% load('gpt2dx.mat')
% 
% %% only need xi, x2
%%xi = gpt2dx.dx.xi;
%%x2 = gpt2dx.dx.x2;
%%psi = gpt2dx.psi;


%% figure
scale = 2.4;

% Set up the figure and colormap
fig = figure('Units', 'Inches', 'Position', [0, 0, 5.5, 4.5] * scale, ...
                 'PaperPositionMode', 'auto', 'Resize', 'on');

set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, 5.5, 4.5] * scale); % originally 5.5

fontsize = 16;

% Define dimensions and time points
dim1 = 2;
dim2 = 3;
t1s = [12 14 16 18];
t2s = [13 15 17 19 21];

% Create a tiled layout
tiledlayout(length(t2s), length(t1s), 'TileSpacing', 'compact', 'Padding', 'compact');

% Loop through time points and create plots
for idx1 = 1:length(t1s)
    t1 = t1s(idx1);
    u1 = psi{t1}(:, dim1);
    u2 = psi{t1}(:, dim2);

    for idx2 = idx1:length(t2s)
        t2 = t2s(idx2);

        % Determine the appropriate tile
        subidx = sub2ind([length(t1s), length(t2s)], idx1, length(t2s) - idx2 + 1);

        % Select the next tile for the plot and get the current axes handle
        ax = nexttile(subidx);

        % Plot the data
        plot(xi{t1, t2}' * u1, xi{t1, t2}' * u2, '.');
        hold on;

        % Add scatter plot with transparency
        scatter(x2{t1, t2}' * u1, x2{t1, t2}' * u2, 10, 'filled', 'MarkerFaceColor', 0.5 * ones(3, 1), 'MarkerFaceAlpha', 0.5);
        
        % Set axis properties for current subplot
        axis equal;
        set(ax, 'XTick', [], 'YTick', []);  % Remove x-ticks and y-ticks using axes handle

        % Label left column with t2
        if idx1 == 1
            ylabel(['$t+\tau=$' num2str(t2)], 'Interpreter', 'latex', 'FontSize', fontsize);
        end
        if idx1 == idx2
            xlabel(['$t=$' num2str(t1)], 'Color', 'k', 'Interpreter', 'latex', 'FontSize', fontsize);
        end
    end

    % % Add x-labels only to the bottom row of the tiled layout
    % ax = nexttile(sub2ind([length(t1s), length(t2s)], idx1, length(t2s)));  % Target the last row in the current column
    % xlabel(['$t=$' num2str(t1)], 'Color', 'k', 'Interpreter', 'latex', 'FontSize', fontsize);
    % set(ax, 'XTick', [], 'YTick', []);  % Remove x-ticks for the x-label subplot
    % if idx1 > 1
    % ax.XColor = 'white';
    % ax.YColor = 'white';
    % end
    % xlabel(['$t=$' num2str(t1)], 'Color', 'k', 'Interpreter', 'latex', 'FontSize', fontsize);
end

%%
%exportgraphics(gcf, 'fig-extrapolation-v07.png', 'ContentType', 'image', 'Resolution', 300);
end

