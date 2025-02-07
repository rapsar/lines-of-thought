function [] = fig02(svc,svl,partial_softmax)

%% figure parameters
scale = 2.4;

% Set up the figure and colormap
fig = figure('Units', 'Inches', 'Position', [0,0, 5.5, 1.5]*scale, ...
                 'PaperPositionMode', 'auto', 'Resize', 'on');

set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, 5.5, 1.5]*scale);

fontsize = 14;

%% realign singular vectors
psi = [];

psi{1} = svc{1};

for t = 2:24

    [order, signs] = compute_order_and_signs(psi{t-1}, svc{t});
    psi{t} = svc{t}(:, order) .* signs;

end

%% angles between singular vectors
ax = subplot(1,3,1);

for i = 1:4
    a = cellfun(@(x) x(:,i),psi,'UniformOutput',false);
    a = horzcat(a{:});
    p = pdist2(a',a','cosine');
    P{i} = acos((abs(1-p)))*180/pi;
end

imagesc([P{1} ones(24,1) P{2} ; ones(1,49) ; P{3} ones(24,1) P{4}])
axis equal
colormap(ax,'parula')
clim([0 90])

xlim([0.5 48.5]);
ylim([0.5 48.5]);

hold on
xline(24.5, 'k', 'LineWidth', 2);  % 'k' specifies black color
yline(24.5, 'k', 'LineWidth', 2);  % 'k' specifies black co

cb = colorbar;
cb.Ticks = [0 45 90];
cb.FontSize = fontsize;

xlabel('$t_1$', 'Interpreter', 'latex','FontSize',fontsize);
ylabel('$t_2$', 'Interpreter', 'latex','FontSize',fontsize);

box on
xticks([])
yticks([])

%% singular values
subplot(1,3,2);

cmap = jet(24);
for i=1:24
    loglog(2:1024,svl{i}(2:end),'Color',cmap(i,:))
    hold on
end
xlabel('S.Val. rank','FontSize',fontsize)
ylabel('S.Val. magnitude','FontSize',fontsize)
xlim([1 512])

%% KL divergence for various cutoffs
subplot(1,3,3);

cutoffs = [2 4 8 16 32 64 128 256 384 512 640 768 896 1024];

for i=1:length(cutoffs)
    kd(i,:) = kl_divergence(partial_softmax(:,:,i), partial_softmax(:,:,end));
end

plot(cutoffs,mean(kd,2),'k.-')
xticks([128 256 512 1024])

hold on
kd0 = kl_divergence(partial_softmax(:,1:end-1,end), partial_softmax(:,2:end,end));
yline(median(kd0),'r--')

xlabel('$K$', 'Interpreter', 'latex','FontSize',fontsize);
ylabel('$\langle \mathrm{KL} ( \mathbf{p}^\mathcal{V}_K \Vert \mathbf{p}^\mathcal{V} ) \rangle$', 'Interpreter', 'latex','FontSize',fontsize);

end

function Dkl = kl_divergence(p, q)

    % Replace zeros to avoid log(0)
    p(p == 0) = eps; 
    q(q == 0) = eps;

    % Compute KL divergence
    Dkl = sum(p .* log(p ./ q),1);
end


function [order, signs] = compute_order_and_signs(R0, R1)
    % compute_order_and_signs Reorder and sign-flip columns of R1 to match R0
    %
    % Args:
    %   R0: Reference orthonormal matrix (m x n)
    %   R1: Orthonormal matrix to reorder (m x n)
    %
    % Returns:
    %   order: 1 x n array indicating the new order of columns in R1
    %   signs: 1 x n array indicating the sign flip for each column in R1

    [m, rank] = size(R0);
    order = -1 * ones(1, rank);
    signs = ones(1, rank);
    
    used = false(1, rank);
    
    for i = 1:rank
        basemode = R0(:, i);
        maxval = -1;
        maxidx = -1;
        for j = 1:rank
            if ~used(j)
                current = abs(basemode' * R1(:, j)); % Magnitude of inner product
                if current >= maxval
                    maxidx = j;
                    maxval = current;
                end
            end
        end
        if maxidx == -1
            error('No valid ordering of modes found');
        end
        order(i) = maxidx;
        used(maxidx) = true;
    end
    
    % Check if ordering is a permutation
    if length(unique(order)) ~= rank
        error('No valid ordering of modes found');
    end
    
    % Determine signs
    for i = 1:rank
        dot_product = R0(:, i)' * R1(:, order(i));
        if dot_product < 0
            signs(i) = -1;
        end
    end
end



