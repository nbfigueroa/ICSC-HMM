% function [stateSeq INDS Nr] = sample_r(data_struct,dist_struct,theta,stateSeq,INDS)
% Sample the sequence of measurement noise mixture component assignments
% given the observation sequence (y_t - C*x_t) and the dynamic parameters.

function [stateSeq INDS Nr] = sample_r(data_struct,dist_struct,theta,stateSeq,INDS)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define and initialize parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define parameters:
pi_r = dist_struct.pi_r;

Kr = length(pi_r);

% Initialize count matrix with Nr(i) = # of obs assigned to mix comp i:
Nr = zeros(1,Kr);

% Preallocate INDS
for ii = 1:length(data_struct)
  T = size(data_struct(ii).tildeY,2);
  INDS(ii).obsIndr(1:Kr) = struct('inds',sparse(1,T),'tot',0);
end

for ii=1:length(data_struct)
    
    if Kr==1  % if single Gaussian measurement noise model, set r to 1's
        
        r = ones(1,T);
        
    else
        
        T = size(data_struct(ii).tildeY,2);
         
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute likelihoods and messages %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        temp_data_struct.obs = data_struct(ii).tildeY;
        
        % Compute likelihood(kr,u_i) of each observation u_i under each
        % parameter theta(kr):
        likelihood = compute_likelihood(temp_data_struct,theta.theta_r,'Gaussian',Kr,1);
        likelihood = squeeze(likelihood);
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Sample the state sequence %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Sample r(1),...,r(T).      
       
        % Multiply likelihood(kr,u_i) by prior p_r(kr):
        Pr = pi_r(ones(1,T),:)'.*likelihood;
        Pr = cumsum(Pr);
        Pr_end = Pr(end,:).*rand(1,T);
        r = 1 + sum(Pr_end(ones(1,size(Pr,1)),:)>Pr,1);
        
    end
    
    % Add counts from sequence ii to count matrix:
    Nr = Nr + histc(r,[1:Kr]);
    
    % Store newly sample mixture component sequence:
    stateSeq(ii).r = r;
    
    % Build up structure containing indices t where y_t has r_t = kr:
    for kr=1:Kr
        INDS(ii).obsIndr(kr).inds(1:Nr(kr)) = find(r==kr);
        INDS(ii).obsIndr(kr).tot = Nr(kr);
    end

end

return;
