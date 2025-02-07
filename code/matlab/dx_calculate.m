function [dxstat] = dx_calculate(traj,phi,sig,keep_dx,dim)
% Calculate differences between linearly extrapolated and true positions
% traj  nDim x nLayers x nTraj array of trajectories
% phi   singular vectors (cell array)
% sig   singular values
% keep  if true keep dx, otherwise, only moments
% dim   truncate analysis to a number of dimensions  

warning('Make sure traj contains only layers output, not embeddings or layernorm outputs.')

spaceDim = min(size(traj,1),size(traj,3));
nTimepoints = size(traj,2);

if nargin < 4
    keep_dx = false;
    dim = spaceDim;
elseif nargin < 5
    dim = spaceDim;
end

if nTimepoints ~= length(phi)
    error('Trajectories and singular bases must have same length')
end

for t1 = 1:nTimepoints

    for t2 = (t1+1):nTimepoints 

        M1 = phi{t1}(:,1:dim); % singular vectors at layer1
        M2 = phi{t2}(:,1:dim); % singular vectors at layer2

        if isvector(sig{t1}) % if sig comes from pca()
            S1 = sig{t1}(1:dim);
            S2 = sig{t2}(1:dim);
            LM = sqrt(diag(S2./S1)); % need sqrt because pca() returns variances

        else
            S1 = sig{t1}(1:dim,1:dim);
            S2 = sig{t2}(1:dim,1:dim);
            LM = diag(S2)./diag(S1);
            LM = diag(LM);
        end

        x1{t1} = squeeze(traj(:,t1,:));
                
        % extrapolating based on rotation and stretch between t1 and t2
        xi{t1,t2} = M2 * LM * (M1' * x1);
        
        % true final location
        x2{t2} = squeeze(traj(:,t2,:));
    
        % difference between extrapolated and real
        dx{t1,t2} = x2{t2} - xi{t1,t2};

        % calculate statistics
        dxstat.m{t1,t2} = mean(dx{t1,t2},2); % mean
        dxstat.v{t1,t2} = var(dx{t1,t2},[],2); % variance
        dxstat.k{t1,t2} = kurtosis(dx{t1,t2},0,2); % kurtosis

        if ~keep_dx

            x1{t1} = [];
            xi{t1,t2} = [];
            x2{t2} = [];
            dx{t1,t2} = [];

        end

    end

    w = waitbar(t1/nTimepoints);

end

close(w)

dxstat.x1 = x1;
dxstat.x2 = x2;
dxstat.xi = xi;
dxstat.dx = dx;

end



