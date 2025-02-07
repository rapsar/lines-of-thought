function xLgv = LoT_langevin_integration(traj,t1,t2,phi,sig,alpha,lambda)
%LOT_LANGEVIN_INTEGRATION Extrapolate LoTs by integration of the Langevin model 
%   

%% can change
dt = 0.1;
nSamples = 100;

%%
nTraj = size(traj,3);
nDim = size(traj,1);

param.phi = phi;
param.sig = sig;
param.alpha = alpha;
param.lambda = lambda;

%% compute matrices

t = t1:dt:t2;

for k = 1:length(t)-1
    A{k} = compute_A(t(k),param);
    B{k} = compute_B(t(k),param);
    w = waitbar(k/length(t));
end

close(w)

%% integrate
system('caffeinate -dims &');

xLgv = zeros(nDim,(t2-t1)+1,nTraj,nSamples);

for j = 1:nTraj

    x0 = traj(:,t1,j);

    for s = 1:nSamples

        xLgv(:,t1,j,s) = x0;
        xold = x0;

        for k = 1:length(t)-1

            % Brownian increments
            dW = sqrt(dt) * randn(nDim, 1); 
            xnew = xold + A{k} * xold * dt + B{k} * dW;
            xold = xnew;
            
            tk = t(k+1);
            if abs(tk-round(tk)) < dt
                xLgv(:,round(tk),j,s) = xnew;
            end
        end
        
    end

w = waitbar(j/nTraj);

end

close(w)

system('killall caffeinate');

end


%% intermediate functions


function A = compute_A(t,param)

    phi = param.phi;

    t1 = floor(t);
    t2 = t1+1;

    [R,dR] = interpMdM(phi{t1}, phi{t2}, t1, t2, t);

    dS = get_dS(t,param); 

    A = dR * R' + R * dS * R';

end

function B = compute_B(t,param) 
    
    alpha = param.alpha;
    lambda = param.lambda;

    B = sqrt(alpha * lambda * exp(lambda * t)) * eye(1024);

end

function [R, dR] = interpMdM(M1, M2, t1, t2, t)
    % interpMdM interpolates between M1 and M2 at time t
    % and computes the derivative dM/dt at time t.
    %
    % Inputs:
    %   M1 - Orthogonal matrix at time t1
    %   M2 - Orthogonal matrix at time t2
    %   t1 - Start time
    %   t2 - End time
    %   t  - Query time, t1 <= t <= t2
    %
    % Outputs:
    %   M_t   - Interpolated matrix at time t
    %   dM_dt - Derivative dM/dt at time t

    % Validate inputs
    if t < t1 || t > t2
        error('Interpolation parameter t must be within [t1, t2].');
    end

    % Compute the relative rotation matrix
    M_rel = M1' * M2;

    % Compute the matrix logarithm of the relative rotation
    A = logm(M_rel);

    % Compute the interpolation factor alpha
    alpha = (t - t1) / (t2 - t1);

    % Compute the interpolated matrix using the matrix exponential
    M_interp = M1 * expm(alpha * A);
    
    % Output
    R = M_interp;

    % Compute the derivative dM/dt
    dAlpha_dt = 1 / (t2 - t1);
    dR = dAlpha_dt * M_interp * A;
    
end

function dS = get_dS(t,param)
    % get_dS computes the diagonal matrix S_dot(t) where
    % S_dot(t) = diag(d/dt sqrt(sigma_i(t)) / sqrt(sigma_i(t)))
    %
    % Inputs:
    %   t              - Scalar or vector of time points within [1, 24].
    %   singular_values - 24 x 1024 matrix, where each row corresponds to t=1,...,24,
    %                    and each column corresponds to a singular value sigma_i.
    %
    % Output:
    %   S_dot - If t is scalar, a 1024x1024 diagonal matrix.
    %           If t is a vector, a 1024x1024xlength(t) array of diagonal matrices.

    singular_values = cell2mat(param.sig);
   
    
    % % Validate Inputs
    % if any(t < 1) || any(t > 24)
    %     error('Input t must be within the range [1, 24].');
    % end
    % 
    % [numTimeSteps, numSingularValues] = size(singular_values);
    % if numTimeSteps ~= 24
    %     error('singular_values must have 24 rows, corresponding to t=1,...,24.');
    % end
    
    % Define the original time points
    original_t = 1:length(sig);
    
    % Create cubic spline interpolant for singular values
    % MATLAB's spline function can handle matrix inputs, interpolating each column separately
    pp = spline(original_t, singular_values); % Piecewise polynomial structure
    
    % Compute the derivative of the spline interpolant
    pp_der = fnder(pp); % Derivative of the piecewise polynomial
    
    % Evaluate singular values at desired t
    sigma_t = ppval(pp, t); % Size: 1x1024 (if t is scalar) or length(t)x1024
    sigma_dot = ppval(pp_der, t); % Same size as sigma_t
    
    % Compute the entries for S_dot
    % To avoid division by zero, introduce a small epsilon
    epsilon = 1e-12;
    sigma_t_safe = max(sigma_t, epsilon); % Ensure no sigma_i(t) is below epsilon
    
    % Compute d/dt sqrt(sigma_i(t)) / sqrt(sigma_i(t)) = (sigma_dot / (2*sqrt(sigma))) / sqrt(sigma)
    % Simplifies to sigma_dot / (2 * sigma)
    S_dot_entries = sigma_dot ./ (2 .* sigma_t_safe); % Element-wise division
    
    % % Construct diagonal matrices
    % if isscalar(t)
    %     % If t is a scalar, create a single diagonal matrix
    %     S_dot = diag(S_dot_entries);
    % else
    %     % If t is a vector, create a 3D array of diagonal matrices
    %     numQueries = length(t);
    %     S_dot = zeros(numSingularValues, numSingularValues, numQueries);
    %     for idx = 1:numQueries
    %         S_dot(:, :, idx) = diag(S_dot_entries(idx, :));
    %     end
    % end

    S_dot = diag(S_dot_entries);
    dS = S_dot;
end
