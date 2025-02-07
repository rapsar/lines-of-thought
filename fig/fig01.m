function [] = fig01(traj, pvc)

u1 = pvc{24}(:,1);
u2 = pvc{24}(:,2);
u3 = pvc{24}(:,3);

scale = 2.2;

% Set up the figure and colormap
fig = figure('Units', 'Inches', 'Position', [0, 0, 5.5, 2.5] * scale, ...
                 'PaperPositionMode', 'auto', 'Resize', 'off');
set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, 5.5, 2.5] * scale);

fontsize = 16;

% Create a tiled layout with 1 row and 2 columns
tiledlayout(1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% First subplot (left)
ax1 = nexttile;
hold on;
colormap(jet);  % Jet colormap (from blue to red)

N = 300;
for i = 1:N
    matrix = traj(:,3:26,i);

    for t = 1:23
        x = u1' * [matrix(:,t), matrix(:,t+1)];
        y = u2' * [matrix(:,t), matrix(:,t+1)];
        z = u3' * [matrix(:,t), matrix(:,t+1)];
        c = [t, t+1];

        % Use surf to plot the line with smooth color transition
        surf([x; x], [y; y], [z; z], [c; c], ...
            'EdgeColor', 'interp', 'FaceColor', 'none', 'LineWidth', 1);
    end
end

xlabel('$\mathbf{u}_1$', 'Interpreter', 'latex', 'FontSize', fontsize, 'Rotation', 0);
ylabel('$\mathbf{u}_2$', 'Interpreter', 'latex', 'FontSize', fontsize, 'Rotation', 0);
zlabel('$\mathbf{u}_3$', 'Interpreter', 'latex', 'FontSize', fontsize, 'Rotation', 0);
grid on;
hold off;
axis equal off;
view([-10, 55]);
xticks([0 500 1000]);
yticks([-500 0 500]);

% Second subplot (right)
ax2 = nexttile;
mtk = mean(traj(:,3:26,:),3);
stk = std(traj(:,3:26,:),0,3);

ts = 12:2:24;
cmap = parula(24);
for i = ts
    trj = squeeze(traj(:,i+2,:));
    plot3(trj(269,:), trj(219,:), trj(591,:), '.', 'Color', cmap(i,:));
    hold on;
end
axis equal;
grid on;

gr = 0.8;
for i = ts
    trj = squeeze(traj(:,i+2,:));
    plot3(0 * trj(269,:) + 50, trj(219,:), trj(591,:), '.', 'Color', gr * ones(1,3));
    plot3(trj(269,:), 0 * trj(219,:) + 400, trj(591,:), '.', 'Color', gr * ones(1,3));
    plot3(trj(269,:), trj(219,:), 0 * trj(591,:) - 50, '.', 'Color', gr * ones(1,3));
end

xlabel('$\mathbf{e}_{269}$', 'Interpreter', 'latex', 'FontSize', fontsize, 'Rotation', 0);
ylabel('$\mathbf{e}_{219}$', 'Interpreter', 'latex', 'FontSize', fontsize, 'Rotation', 0);
zlabel('$\mathbf{e}_{591}$', 'Interpreter', 'latex', 'FontSize', fontsize, 'Rotation', 0);
xlim([-650, 50]);
ylim([-200, 400]);
zlim([-50, 650]);
view([-41, 27]);
zticks([0 200 400 600]);

%exportgraphics(gcf, 'myfigure.pdf', 'ContentType', 'vector', 'Resolution', 300);

end

