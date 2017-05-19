% Given a previously sampled mode sequence, model parameters, and backward
% filter parameters, sequentially sample the state sequence.

function internal_data_struct = sample_x(data_struct,stateSeq,theta,Lambda_t_t_bwd,theta_t_t_bwd,C,invP0);

% Define constancts:
dimx = size(theta.A,1); % dimension of the continuous state
A = theta.A; % dynamic or "propagation" matrix of state
invSigma = theta.invSigma;
if isfield(theta,'mu')
    mu = theta.mu; % structure of mode-specific mean of Gaussian process noise
    invSigmaA = zeros(size(invSigma));
    invSigmaMu = zeros(size(mu));
    Kz = size(A,3);
    for kz=1:Kz
        invSigmaA(:,:,kz)  = invSigma(:,:,kz)*A(:,:,kz);
        invSigmaMu(:,kz) = invSigma(:,:,kz)*mu(:,kz);
    end
else
    invSigmaA = zeros(size(invSigma));
    invSigmaMu = zeros(size(A,1),size(A,3));
    Kz = size(A,3);
    for kz=1:Kz
        invSigmaA(:,:,kz)  = invSigma(:,:,kz)*A(:,:,kz);
    end
end

% Define an "internal" data structure whose "observations" will be the
% sampled state sequence (and order 1 VAR process):
internal_data_struct = data_struct;

for ii=1:length(data_struct)

    T = size(data_struct(ii).obs,2);
    
    % Define z_0 = 0 as a place holder since the state sequence is defined
    % by x_{0:T}:
    z_ii = [0 stateSeq(ii).z];

    % Sample x_t from t=1:T (use P0 as initial covariance or p(x_0)):
    x = zeros(dimx,T+1);
    Sigma_1 = inv(Lambda_t_t_bwd(:,:,1)+invP0);
    mu_1 = Sigma_1*theta_t_t_bwd(:,1);
    x(:,1) = mu_1 + chol(Sigma_1)'*randn(dimx,1);
    for t=2:T+1
        % Combine p(x_t|x_{t-1},z_t) with updated backward message at time
        % t:
        Lambda_tot = invSigma(:,:,z_ii(t),1)+Lambda_t_t_bwd(:,:,t);
        theta_tot =  invSigmaA(:,:,z_ii(t),1)*x(:,t-1)+invSigmaMu(:,z_ii(t),1) + theta_t_t_bwd(:,t);
        Sigma_t = inv(Lambda_tot);
        Sigma_t = 0.5*(Sigma_t + Sigma_t');
        mu_t = Sigma_t*theta_tot;
        x(:,t) = mu_t + chol(Sigma_t)'*randn(dimx,1);
    end

    %
    [X,valid] = makeDesignMatrix(x,1);

    internal_data_struct(ii).obs = x(:,find(valid));
    internal_data_struct(ii).X = X(:,find(valid));
    internal_data_struct(ii).blockSize = ones(1,size(internal_data_struct(ii).obs,2));
    internal_data_struct(ii).blockEnd = cumsum(internal_data_struct(ii).blockSize);
    if isfield(data_struct(ii),'true_labels')
        internal_data_struct(ii).true_labels = data_struct(ii).true_labels;
    end
    internal_data_struct(ii).tildeY = (data_struct(ii).obs-C*x(:,find(valid)));
    
end
internal_data_struct(1).test_cases = data_struct(1).test_cases;

return;