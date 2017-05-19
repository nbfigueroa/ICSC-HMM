function [hyperparams] = sample_hyperparams_init(stateCounts,hyperparams,hyperhyperparams,HMMmodelType,resample_kappa)

% Hyperparams for gamma dist over \alpha+\kappa, where transition distributions
% \pi_j \sim DP(\alpha+\kappa,(\alpha\beta+\kappa\delta(j))/(\alpha+\kappa))
% = DP(alpha_p_kappa, (1-\rho)*\beta + \rho*\delta(j)):
a_alpha=hyperhyperparams.a_alpha;
b_alpha=hyperhyperparams.b_alpha;

% Hyperparams for beta dist over \rho, where \rho relates \alpha+\kappa to
% \alpha and \kappa individually.
c=hyperhyperparams.c;
d=hyperhyperparams.d;

Ns = stateCounts.Ns;

switch HMMmodelType
    
    case 'HDP'
        
        % Hyperparams for gamma dist over \gamma, where avg transition distribution
        % \beta \sim stick(\gamma):
        a_gamma=hyperhyperparams.a_gamma;
        b_gamma=hyperhyperparams.b_gamma;
        
        % Resample concentration parameters:
        alpha0_p_kappa0 = a_alpha / b_alpha;    % Gj concentration parameter
        gamma0 = a_gamma / b_gamma;    % G0 concentration parameter
        
        hyperparams.gamma0 = gamma0;
        
    case 'finite'
        %         alpha0_p_kappa0 = alpha0_p_kappa0;
        % Resample concentration parameters:
        alpha0_p_kappa0 = a_alpha / b_alpha;    % Gj concentration parameter
end

if size(Ns,2)>1  % MAYBE CHANGE THIS WHEN HDP-HMM with FINITE EMISSION STUFF IS ADDED
    
    % Hyperparams for gamma dist over \sigma, where HMM-state-specific mixture
    % weights \psi_j \sim stick(\sigma):
    a_sigma=hyperhyperparams.a_sigma;
    b_sigma=hyperhyperparams.b_sigma;
    
    sigma0 = a_sigma / b_sigma;
else
    sigma0 = 1;
end

if resample_kappa
    rho0 = c/(c+d);
else
    rho0 = 0; % betarnd(0,1);
end

if isfield(stateCounts,'Nr')
    
    if length(stateCounts.Nr)>1  % Only spend time resampling eta0 if MoG measurement noise
        
        eta0 = hyperparams.eta0;
        
        % Hyperparams for gamma dist over \sigma, where HMM-state-specific mixture
        % weights \psi_j \sim stick(\sigma):
        a_eta=hyperhyperparams.a_eta;
        b_eta=hyperhyperparams.b_eta;
        
        eta0 = a_eta / b_eta;
        
    else
        eta0 = 1;
    end
    
    hyperparams.eta0 = eta0;
    
end

hyperparams.alpha0_p_kappa0 = alpha0_p_kappa0;
hyperparams.sigma0 = sigma0;
hyperparams.rho0 = rho0;
