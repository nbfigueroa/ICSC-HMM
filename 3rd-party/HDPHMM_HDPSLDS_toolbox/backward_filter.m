% function [Lambda_t_t_bwd theta_t_t_bwd] = backward_filter(stateSeq,data_struct,theta,C)
% Given a fixed mode sequence and model parameters, run the backward filter 

function [Lambda_t_t_bwd theta_t_t_bwd] = backward_filter(stateSeq,data_struct,theta,C)

% Define constants:
dimx = size(theta.A,1);
dimy = size(data_struct(1).obs,1);
Kz = size(theta.A,3);
Kr = size(theta.theta_r.invSigma,3);

% Get parameters:
A = theta.A;  % dynamic or "propagation" matrix of state
if isfield(theta,'mu')
    mu = theta.mu;  % structure of mode-specific mean of Gaussian process noise
else
    mu = zeros(dimx,Kz);
end
invSigma = theta.invSigma;  %  structure of mode-specific inverse covariance matrix of Gaussian process noise
Lambda_r = theta.theta_r.invSigma;  % structure of mixture-specific inverse covariance matrix of MoG measurement noise
if isfield(theta.theta_r,'mu')
    mu_r = theta.theta_r.mu; % structure of mixture-specific mean of MoG measurement noise
else
    mu_r = zeros(dimy,Kr);
end

Sigma = zeros(size(invSigma));
for kz=1:Kz
    Sigma(:,:,kz) = inv(invSigma(:,:,kz));
end

for ii=1:length(data_struct)
        
    T = size(data_struct(ii).obs,2);
    
    % Define y_0 = z_0 = r_0 = 0.  These terms will not be used, just place
    % holders since the state sequence is defined by x_{0:T}:
    z_ii = [0 stateSeq(ii).z];
    r_ii = [0 stateSeq(ii).r];
    y = [zeros(dimy,1) data_struct(ii).obs-mu_r(:,r_ii(2:end))];
    

    %% RUN BACKWARD FILTER %%
    
    % Initialize structures:
    Lambda_t_t_bwd = zeros(dimx,dimx,T+1);
    theta_t_t_bwd = zeros(dimx,T+1);
    
    % Define initial conditions:
    Lambda_t_t_bwd(:,:,T+1) = C'*Lambda_r(:,:,r_ii(T+1))*C;
    theta_t_t_bwd(:,T+1) = C'*Lambda_r(:,:,r_ii(T+1))*y(:,T+1);
    for t=T+1:-1:2
        
        % Define special terms for backward filter given z_t:
        J_t = Lambda_t_t_bwd(:,:,t)*inv(Lambda_t_t_bwd(:,:,t)+invSigma(:,:,z_ii(t)));
        L_t = (eye(size(J_t,1)) - J_t);
        Sigma_t = L_t*Lambda_t_t_bwd(:,:,t)*L_t' + J_t*invSigma(:,:,z_ii(t))*J_t';

        % Propagate the backward message:
        Lambda_t_tm1 = A(:,:,z_ii(t))'*Sigma_t*A(:,:,z_ii(t));
        theta_t_tm1 = A(:,:,z_ii(t))'*L_t*(theta_t_t_bwd(:,t)-Lambda_t_t_bwd(:,:,t)*mu(:,z_ii(t)));
        % Update the backward message with the current observation:
        if t==2
            Lambda_t_t_bwd(:,:,t-1) = Lambda_t_tm1;
            theta_t_t_bwd(:,t-1) = theta_t_tm1;
        else
            Lambda_t_t_bwd(:,:,t-1) = Lambda_t_tm1 + C'*Lambda_r(:,:,r_ii(t-1))*C;
            theta_t_t_bwd(:,t-1) = theta_t_tm1+C'*Lambda_r(:,:,r_ii(t-1))*y(:,t-1);
        end

    end
    
end
