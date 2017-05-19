% Treat each observation as if it were the last in the chain %

function [stateSeq Lambda_t_t_bwd theta_t_t_bwd] = sample_z_seq_init(data_struct,dist_struct,theta,C,invP0)

% Define constants:
dimx = size(theta.A,1); % dimension of state vector
dimy = size(data_struct(1).obs,1); % dimension of observation vector
pi_z = dist_struct.pi_z;  % transition matrix
pi_init = dist_struct.pi_init;  % initial transition distribution
pi_r = dist_struct.pi_r;  % mixture weights for measurement noise MoG model
Kz = size(pi_z,1);  % truncation level of transition distribution
Kr = length(pi_r);  % truncation level of measurement noise DPMM (1 if single Gaussian noise model)

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
    
    % Define y_0 = 0.  This term will not be used, just a place
    % holder since the state sequence is defined by x_{0:T}:
    y = [zeros(dimy,1) data_struct(ii).obs];    
    
    % If no initial mode sequence is provided, sample the mode sequence by
    % sequentially sampling z_t treating it as the last time point (i.e.,
    % as if no future observations existed.)
    if ~isfield(data_struct,'z_init')

        z_ii = zeros(1,T+1);
        r_ii = zeros(1,T+1);

        %% RUN FORWARD FILTER %%

        % Initialize structures:
        Lambda_t_t_fwd = zeros(dimx,dimx,T+1);
        theta_t_t_fwd = zeros(dimx,T+1);

        % Define initial conditions:
        Lambda_t_t_fwd(:,:,1) = invP0;

        % Run filter going forward conditioned on z_{1:T}:
        for t=2:T+1

            marg = zeros(Kr,Kz);
            
            % Compute marginal likelihood term for each possible mixture 
            % component of the measurement noise r_t:
            for rr=1:Kr
                
                % Since we are treating each observation y_t as if it were the
                % last, the backward message from t+1 to t is just
                % N^{-1}(0,0) so that the updated backward message at time
                % t is: 
                Lambda_bwd_msg_r = C'*Lambda_r(:,:,rr)*C;
                theta_bwd_msg_r = C'*Lambda_r(:,:,rr)*(y(:,t)-mu_r(:,rr));
                        
                % Grab the previously computed updated forward message:
                invLambda_t_t_fwd = inv(Lambda_t_t_fwd(:,:,t-1));
                mu_t_t_fwd = invLambda_t_t_fwd * theta_t_t_fwd(:,t-1);
                
                % Compute likelihood for each possible current mode z_t:
                for kk=1:Kz
                    % Compute forward message from t-1 to t assuming
                    % z_t=kk:
                    Lambda_msgk = inv(A(:,:,kk)*invLambda_t_t_fwd*A(:,:,kk)' + Sigma(:,:,kk));
                    theta_msgk  = Lambda_msgk*(mu(:,kk) + A(:,:,kk)*mu_t_t_fwd);
                    
                    % Combine backward message with this kk-specific forward
                    % message:
                    Lambda_k = Lambda_bwd_msg_r + Lambda_msgk;
                    theta_k  = theta_bwd_msg_r + theta_msgk;
                    
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
            
            if t==2
                % If initial time step, compine likelihood with prior
                % probability of choosing z_1 = kk and r_1 = rr:
                Pz = (pi_r'*pi_init).*marg;
            else
                % Otherwise, compine likelihood with prior probability of
                % transitioning from z_{t-1} to z_t = kk. Since we are
                % assuming t is the last time step, there is no z_t = kk to
                % z_{t+1} transition. Also include prior probability of 
                % choosing r_t = rr:
                Pz = (pi_r'*pi_z(z_ii(t-1),:)).*marg;
            end
            
            % Sample (z_t,r_t):
            Pz = Pz(:);
            Pz   = cumsum(Pz);
            ind_zr = 1 + sum(Pz(end)*rand(1) > Pz);
            [r_ii(t) z_ii(t)] = ind2sub([Kr Kz],ind_zr);

            % Propagate the forward message:
            Lambda_tm1_t = inv(A(:,:,z_ii(t))*invLambda_t_t_fwd*A(:,:,z_ii(t))' + Sigma(:,:,z_ii(t)));
            theta_tm1_t  = Lambda_tm1_t*(mu(:,z_ii(t)) + A(:,:,z_ii(t))*mu_t_t_fwd);

            % Update the forward message with the current observation:
            Lambda_t_t_fwd(:,:,t) = Lambda_tm1_t + C'*Lambda_r(:,:,r_ii(t))*C;
            theta_t_t_fwd(:,t) = theta_tm1_t + C'*Lambda_r(:,:,r_ii(t))*(y(:,t)-mu_r(:,r_ii(t)));

        end

    else
        % Set the mode sequence to the specified initial mode sequence:
        z_ii = [0 data_struct(ii).z_init];
        r_ii = [0 ones(1,T)];
        display('Do you want r_{1:T} all ones or randomly initialized?')
        
    end

    % Store sampled mode sequences:
    stateSeq(ii).z = z_ii(2:end);
    stateSeq(ii).r = r_ii(2:end);

    %% CALCULATE BACKWARDS MESSAGES %%

    % Initialize structures:
    Lambda_t_t_bwd = zeros(dimx,dimx,T+1);
    theta_t_t_bwd = zeros(dimx,T+1);
    
    % Define initial conditions:
    Lambda_t_t_bwd(:,:,T+1) = C'*Lambda_r(:,:,r_ii(T+1))*C;
    theta_t_t_bwd(:,T+1) = C'*Lambda_r(:,:,r_ii(T+1))*(y(:,T+1)-mu_r(:,r_ii(T+1)));
    for t=T+1:-1:2

        % Define special terms for backward filter given z_t:
        J_t = Lambda_t_t_bwd(:,:,t)*inv(Lambda_t_t_bwd(:,:,t)+invSigma(:,:,z_ii(t)));
        L_t = (eye(size(J_t,1)) - J_t);
        Sigma_t = L_t*Lambda_t_t_bwd(:,:,t)*L_t' + J_t*invSigma(:,:,z_ii(t))*J_t';

        % Propogate the backward message:
        Lambda_t_tm1 = A(:,:,z_ii(t))'*Sigma_t*A(:,:,z_ii(t));
        theta_t_tm1 = A(:,:,z_ii(t))'*L_t*(theta_t_t_bwd(:,t)-Lambda_t_t_bwd(:,:,t)*mu(:,z_ii(t)));
        % Update the backward message with the current observation:
        if t==2
            Lambda_t_t_bwd(:,:,t-1) = Lambda_t_tm1;
            theta_t_t_bwd(:,t-1) = theta_t_tm1;
        else
            Lambda_t_t_bwd(:,:,t-1) = Lambda_t_tm1 + C'*Lambda_r(:,:,r_ii(t-1))*C;
            theta_t_t_bwd(:,t-1) = theta_t_tm1+C'*Lambda_r(:,:,r_ii(t-1))*(y(:,t-1)-mu_r(:,r_ii(t-1)));
        end

    end
end