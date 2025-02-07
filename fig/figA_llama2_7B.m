%% load trajectories
load('traj_llama2_7B_walden_1000.mat')
llama27B.traj = embeddings;
llama27B.traj = llama27B.traj(:,2:end,:); % remove first embeddings
[nDim,nLayer,nTraj] = size(llama27B.traj);

%% find weird outliers
llama27B.outlierThreshold = 10;

for t = 1:nLayer
    llama27B.r{t} = vecnorm(squeeze(llama27B.traj(:,t,:)));
    llama27B.outliers{t} = find(isoutlier(llama27B.r{t},'ThresholdFactor',llama27B.outlierThreshold));
end

%% remove outliers 
llama27B.bad = [66,223,277,330,435,538,669];
llama27B.good = true(nTraj,1);
llama27B.good(llama27B.bad) = false;
llama27B.traj = llama27B.traj(:,:,llama27B.good);

%% calculate singular vectors/values
for t = 1:nLayer
    [llama27B.phi{t},llama27B.sig{t},~] = svd(squeeze(llama27B.traj(:,t,:)),"econ");
    w = waitbar(t/nLayer);
end
close(w)

%% calculate psi from phi
llama27B.psi = phi2psi(llama27B.phi);

%% calculate dx
llama27B.dx = dx_calculate(llama27B.traj,llama27B.psi,llama27B.sig,false);

%% plot figure
fig_llama27B_dx(llama27B.dx)

%% save figure
exportgraphics(gcf, 'figA_llama2_7B.pdf', 'ContentType', 'vector', 'Resolution', 300);

%% functions

function [] = fig_llama27B_dx(dxstat)
% Plot dx statistics for the llama2_7B trajectories (m/s,log(v),k)

scale = 2.4;

% Set up the figure and colormap
fig = figure('Units', 'Inches', 'Position', [0,0, 5.5, 2.5]*scale, ...
                 'PaperPositionMode', 'auto', 'Resize', 'on');

set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, 5.5, 2.5]*scale); 

fontsize = 14;

%% moments
empty = cellfun(@isempty,dxstat.m);

m = dxstat.m;
m(empty) = {NaN(4096,1)};

v = dxstat.v;
v(empty) = {NaN(4096,1)};

k = dxstat.k;
k(empty) = {NaN(4096,1)};


%% mean/std
subplot(1,3,1)
im = imagesc(cellfun(@(x,y) mean(abs(double(x)./double(y))),m,v));
axis equal
h = colorbar('northoutside');
set(im, 'AlphaData', ~empty);
axis square
xlabel('$t + \tau$', 'Interpreter', 'latex','FontSize',fontsize);
ylabel('$t$', 'Interpreter', 'latex','FontSize',fontsize);
yticks(1:4:32)
xticks(2:4:32)
xlim([0.5 32.5])
ylim([0.5 32.5])
clim([0 0.5])

title('$\langle | \mu / \sigma | \rangle$', 'Interpreter', 'latex','FontSize',fontsize);


%% log(var)
subplot(1,3,2) 
im = imagesc(log10(cellfun(@(x) mean(double(x)),v)));
axis equal
h = colorbar('northoutside');
set(im, 'AlphaData', ~empty);
axis square
xlabel('$t + \tau$', 'Interpreter', 'latex','FontSize',fontsize);
ylabel([]); yticks([]);
xticks(2:4:32)
xlim([0.5 32.5])
ylim([0.5 32.5])

title('$\langle \log (\sigma^2) \rangle$', 'Interpreter', 'latex','FontSize',fontsize);

%colormap(gca,'jet');


%% kurtosis
subplot(1,3,3) 
im = imagesc(cellfun(@(x) mean(abs(double(x))),k) - 3);
axis equal
h = colorbar('northoutside');
set(im, 'AlphaData', ~empty);
axis square
xlabel('$t + \tau$', 'Interpreter', 'latex','FontSize',fontsize);
ylabel([]); yticks([]);
xticks(2:4:32)
xlim([0.5 32.5])
ylim([0.5 32.5])
clim([0 0.8])

title('$\langle | \kappa | \rangle$', 'Interpreter', 'latex','FontSize',fontsize);

end
