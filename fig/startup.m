%% load
load('traj_walden_50tk')
traj = traj_walden_50tk;

% calculate principal components
pvc = cell(1,24);
ppj = cell(1,24);
pvl = cell(1,24);
for i=1:24 
    % vectors, projections, values
    [pvc{i},ppj{i},pvl{i}] = pca(squeeze(traj(:,i+2,:))'); % +2 bc embeddings
end

% calculate singular vectors 
svc = cell(1,24);
spj = cell(1,24);
svl = cell(1,24);
for i=1:24 
    % vectors, projection, values
    [svc{i},spj{i},svl{i}] = pca(squeeze(traj(:,i+2,:))','Centered',false); 
end

%% Figure 1
fig01(traj,svc)

%% Figure 2
load('partial_softmax_50tk')
fig02(svc,svl,partial_softmax_50tk)

%% Figure 3
traj = traj(:,3:26,:);
[nDim,nLayer,nTraj] = size(traj);
% calculate SVD
for t = 1:nLayer
    [phi{t},sig{t},~] = svd(squeeze(traj(:,t,:)),"econ");
    w = waitbar(t/nLayer);
end
psi = phi2psi(phi);
dx = dxCalculate(traj,psi,sig,false);

fig03(dx.xi,dx.x2,psi)