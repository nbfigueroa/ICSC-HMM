% function [stateSeq Lambda_t_t_bwd theta_t_t_bwd] = sample_z_seq(stateSeq_prev,data_struct,dist_struct,theta,C,invP0)
% Given a previously sampled mode sequence and model parameters,
% sequentially sample a new mode sequence z_{1:T} (marginalizing the state
% sequence x_{0:T}) and run the backwards filter. If MoG measurement noise 
% model, also get an initial mixture component sequence r_{1:T}.

function [stateSeq Lambda_t_t_bwd theta_t_t_bwd] = sample_z_seq(stateSeq_prev,data_struct,dist_struct,theta,C,invP0)

stateSeq = stateSeq_prev;

% Define constants:
dimx = size(theta.A,1); % dimension of state vector
dimy = size(data_struct(1).obs,1); % dimension of observation vector
pi_z = dist_struct.pi_z;  % transition matrix
pi_init = dist_struct.pi_init;  % initial transition distribution
pi_r = dist_struct.pi_r;  % mixture weights for measurement noise MoG model
Kz = size(pi_z,1);  % truncation level of transition distribution
Kr = length(pi_r);  % truncation level of measurement noise DPMM

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

normtheta    = zeros(dimx,Kz);
normthetaM   = zeros(dimx,Kz);
dcholLambda  = zeros(dimx,Kz);
dcholLambdaM = zeros(dimx,Kz);

for ii=1:length(data_struct)

    T = size(data_struct(ii).obs,2);
    
    % Define y_0 = z_0 = r_0 = 0.  These terms will not be used, just place
    % holders since the state sequence is defined by x_{0:T}:
    y = [zeros(dimy,1) data_struct(ii).obs];
    z_ii = [0 stateSeq_prev(ii).z];
    r_ii = [0 stateSeq_prev(ii).r];

    %% RUN FORWARD FILTER %%
    
    % Initialize backward filter structures:
    Lambda_t_t_bwd = zeros(dimx,dimx,T+1);
    theta_t_t_bwd = zeros(dimx,T+1);
    
    % Initialize forward filter structures:
    Lambda_t_t_fwd = zeros(dimx,dimx,T+1);
    theta_t_t_fwd = zeros(dimx,T+1);

    % Define initial conditions:
    Lambda_t_t_fwd(:,:,1) = invP0;

    % Run filter going forward conditioned on z_{1:T}:
    for t=2:T+1
        % Define special terms for forward filter:
        invLambda_t_t_fwd = inv(Lambda_t_t_fwd(:,:,t-1));
        mu_t_t_fwd = invLambda_t_t_fwd * theta_t_t_fwd(:,t-1);

        % Propagate the forward message:
        Lambda_tm1_t = inv(A(:,:,z_ii(t))*invLambda_t_t_fwd*A(:,:,z_ii(t))' + Sigma(:,:,z_ii(t)));
        theta_tm1_t  = Lambda_tm1_t*(mu(:,z_ii(t)) + A(:,:,z_ii(t))*mu_t_t_fwd);

        % Update the forward message with the current observation:
        Lambda_t_t_fwd(:,:,t) = Lambda_tm1_t + C'*Lambda_r(:,:,r_ii(t))*C;
        theta_t_t_fwd(:,t) = theta_tm1_t + C'*Lambda_r(:,:,r_ii(t))*(y(:,t)-mu_r(:,r_ii(t)));
    end

    %% SEQUENTIALLY SAMPLE (z_T,r_T),(z_{T-1},r_{T-1}),...,(z_1,r_1) and RUN BACKWARD FILTER %%
    
    % Define initial conditions:
    Lambda_t_tm1 = zeros(dimx,dimx);
    theta_t_tm1 = zeros(dimx,1);
    
    for t=T+1:-1:2
        
        % Sample (z_t,r_t):
        marg = zeros(Kr,Kz);
        
        % Compute marginal likelihood term for each possible mixture
        % component of the measurement noise r_t:
        for rr=1:Kr
            
            % Update filter:
            Lambda_t_t_bwd_r = Lambda_t_tm1 + C'*Lambda_r(:,:,rr)*C;
            theta_t_t_bwd_r = theta_t_tm1+C'*Lambda_r(:,:,rr)*(y(:,t)-mu_r(:,rr));
            
            invLambda_t_t_fwd = inv(Lambda_t_t_fwd(:,:,t-1));
            mu_t_t_fwd = invLambda_t_t_fwd * theta_t_t_fwd(:,t-1);
            
            % Compute likelihood for each possible current mode z_t:
            for kk=1:Kz
                % Compute forward message from t-1 to t assuming z_t=kk:
                Lambda_msgk = inv(A(:,:,kk)*invLambda_t_t_fwd*A(:,:,kk)' + Sigma(:,:,kk));
                theta_msgk  = Lambda_msgk*(mu(:,kk) + A(:,:,kk)*mu_t_t_fwd);
                
                % Combine backward message with this kk-specific forward
                % message:
                Lambda_k = Lambda_t_t_bwd_r + Lambda_msgk;
                theta_k  = theta_t_t_bwd_r + theta_msgk;
                
                cholLambda_k    = chol(Lambda_k);
                cholLambda_msgk = chol(Lambda_msgk);
                normtheta(:,kk)    = cholLambda_k' \ theta_k;
                normthetaM(:,kk)   = cholLambda_msgk' \ theta_msgk;
                dcholLambda(:,kk)  = diag(cholLambda_k);
                dcholLambdaM(:,kk) = diag(cholLambda_msgk);
            end
            % Calculate marginal likelihood by integrating over x_t:
            log_marg = - 0.5*sum(normthetaM.^2,1) + 0.5*sum(normtheta.^2,1) ...
                - sum(log(dcholLambda),1) + sum(log(dcholLambdaM),1);
            marg(rr,:) = exp(log_marg - max(log_marg));
        end
        
        % We have now defined likelihood 'marg', where marg(rr,kk) is the
        % marginal likelihood of the data assuming z_t = kk and r_t = rr
        % conditioned on z_\t, r_\t.
        
        if t==2
            % If initial time step, compine likelihood with prior
            % probability of choosing z_1 = kk and r_1 = rr:
            Pz = (pi_r'*(pi_z(:,z_ii(t+1))'.*pi_init)).*marg;
        elseif t==T+1
            % If last time step, compine likelihood with prior
            % probability of transitioning from z_{T-1} to z_T = kk and
            % choosing r_T = rr:
            Pz = (pi_r'*pi_z(z_ii(t-1),:)).*marg;
        else
            % Otherwise, compine likelihood with prior probability of 
            % transitioning from z_{t-1} to z_t = kk and z_t=kk to z_{t+1}.
            % Also include prior probability of choosing r_t = rr:
            Pz = (pi_r'*(pi_z(:,z_ii(t+1))'.*pi_z(z_ii(t-1),:))).*marg;
        end

        % Sample (z_t,r_t):
        Pz = Pz(:);
        Pz   = cumsum(Pz);
        ind_zr = 1 + sum(Pz(end)*rand(1) > Pz);
        [r_ii(t) z_ii(t)] = ind2sub([Kr Kz],ind_zr);
        
        % Update the backward message with the current observation using sampled r_t:
        Lambda_t_t_bwd(:,:,t) = Lambda_t_tm1 + C'*Lambda_r(:,:,r_ii(t))*C;
        theta_t_t_bwd(:,t) = theta_t_tm1+C'*Lambda_r(:,:,r_ii(t))*(y(:,t)-mu_r(:,r_ii(t)));
        
        % Define special terms for backward filter given z_t:
        J_t = Lambda_t_t_bwd(:,:,t)*inv(Lambda_t_t_bwd(:,:,t)+invSigma(:,:,z_ii(t)));
        L_t = (eye(size(J_t,1)) - J_t);
        Sigma_t = L_t*Lambda_t_t_bwd(:,:,t)*L_t' + J_t*invSigma(:,:,z_ii(t))*J_t';
        
        % Propagate the backward message:
        Lambda_t_tm1 = A(:,:,z_ii(t))'*Sigma_t*A(:,:,z_ii(t));
        theta_t_tm1 = A(:,:,z_ii(t))'*L_t*(theta_t_t_bwd(:,t)-Lambda_t_t_bwd(:,:,t)*mu(:,z_ii(t)));
    end
    
    % Create updated message at t=0 as the message from t=1 to t=0 since
    % there is no y_0 to update with. (Can invision as a measurement with
    % infinite uncertainty.)
    Lambda_t_t_bwd(:,:,1) = Lambda_t_tm1;
    theta_t_t_bwd(:,1) = theta_t_tm1;

    % Store sampled mode sequences:
    stateSeq(ii).z = z_ii(2:end);
    stateSeq(ii).r = r_ii(2:end);
    
end
