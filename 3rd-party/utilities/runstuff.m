%%% WARNING: THIS CODE GENERATES DATA FROM THE PRIOR, WHICH DOES NOT ALWAYS
%%% MEAN THAT THE REALIZATIONS HAVE DISTINGUISHABLE DYNAMICAL MODES. THIS
%%% CODE SIMPLY SERVES TO DEMONSTRATE HOW TO RUN THE HDPHMMDPinference
%%% SCRIPT.

function runstuff(caseNumber,trial_vec,saveDir)

d = 2;
T = 2000;

% obsModelType = {'Gaussian','AR','SLDS'};

switch caseNumber
    case 1
        obsModelType = 'Gaussian';
        priorType = 'IW-N';  % prior on Gaussian N(mu_{k,j},Sigma_{k,j}) emissions (non-conjugate)
        sig0 = 10;  % covariance of the N(mu0,sig0) prior on mu_{k,j}
        meanSigma = eye(d);  % expected mean of IW(nu,nu_delta) prior on Sigma_{k,j}
        Kz = 10;  % truncation level of the DP prior on HMM transition distributions pi_k
        Ks = 10;  % truncation level of the DPMM on emission distributions pi_s
        
    case 2
        obsModelType = 'Gaussian';
        priorType = 'NIW'; 
        kappa = 0.1;  % NIW(kappa,theta,delta,nu_delta)
        meanSigma = eye(d);
        Kz = 10;
        Ks = 10;
        
    case 3
        obsModelType = 'AR';
        priorType = 'MNIW';  % prior on 
        r = 2;
        K = inv(diag([0.1*ones(1,d*r)]));
        meanSigma = eye(d);
        Kz = 10;
        
    case 4
        obsModelType = 'AR';
        priorType = 'MNIW-N';
        r = 2;
        K = inv(diag([0.1*ones(1,d*r)]));
        sig0 = 3;
        meanSigma = eye(d);
        Kz = 10;
        
    case 5
        obsModelType = 'AR';
        priorType = 'N-IW-N';
        r = 2;
        K = inv(diag([0.1*ones(1,d*r)]));
        sig0 = 3;
        meanSigma = eye(d);
        Kz = 10;
        
    case 6
        obsModelType = 'AR';
        priorType = 'ARD';
        r = 2;
        meanSigma = eye(d);        
        Kz = 10;
        
    case 7
        obsModelType = 'SLDS';
        dy = 2;
        priorType = 'MNIW';
        meanSigma = eye(d);
        K = inv(diag([0.1*ones(1,d)]));
        y_priorType = 'IW';
        y_var = 1;
        P0 = 1;
        Kz = 10;
        Kr = 10;
        
    case 8
        obsModelType = 'SLDS';
        dy = 1;
        priorType = 'MNIW';
        meanSigma = eye(d);
        K = inv(diag([0.1*ones(1,d)]));
        y_priorType = 'IW-N';
        y_var = 1;
        sig0_y = 1;
        P0 = 1;
        Kz = 10;
        Kr = 10;
        
    case 9
        obsModelType = 'SLDS';
        dy = 1;
        priorType = 'MNIW';
        meanSigma = eye(d);
        K = inv(diag([0.1*ones(1,d)]));
        y_priorType = 'NIW';
        y_var = 1;
        kappa_y = 1;
        P0 = 1;
        Kz = 10;
        Kr = 10;
        
    case 10
        obsModelType = 'SLDS';
        dy = 1;
        priorType = 'N-IW-N';
        K = inv(diag([0.1*ones(1,d)]));
        meanSigma = eye(d);
        sig0 = 10;
        y_priorType = 'IW';
        y_var = 1;
        P0 = 1;
        Kz = 10;
        
    case 11
        obsModelType = 'SLDS';
        dy = 1;
        priorType = 'N-IW-N';
        meanSigma = eye(d);
        K = inv(diag([0.1*ones(1,d)]));
        sig0 = 25;
        y_priorType = 'IW-N';
        y_var = 1;
        sig0_y = 25;
        P0 = 1;
        Kz = 10;
        Kr = 10;
        
end

%%
% Setting for inference:

clear settings

switch obsModelType
    case {'AR','SLDS'}
        settings.Kz = Kz;   % truncation level for mode transition distributions
        settings.Ks = 1;  % truncation level for mode transition distributions
        if strcmp(obsModelType,'SLDS') && ~strcmp(y_priorType,'IW')
            settings.Kr = Kr;  % truncation level for MoG measurement noise
        end
    case 'Gaussian'
        settings.Kz = Kz;   % truncation level for mode transition distributions
        settings.Ks = Ks;  % truncation level for mode transition distributions
end
settings.Niter = 1000;  % Number of iterations of the Gibbs sampler
settings.resample_kappa = 1;  % Whether or not to use sticky model
settings.seqSampleEvery = 100; % How often to run sequential z sampling
settings.saveEvery = 100;  % How often to save Gibbs sample stats
settings.storeEvery = 1;
settings.storeStateSeqEvery = 100;
settings.ploton = 1;  % Whether or not to plot the mode sequence while running sampler
settings.plotEvery = 20;
settings.plotpause = 0;  % Length of time to pause on the plot
settings.saveDir = saveDir;  % Directory to which to save files

%%
% Set Hyperparameters

clear model

% Type of dynamical system:
model.obsModel.type = obsModelType;

if strcmp(obsModelType,'AR')
    % Order of AR process:
    model.obsModel.r = r;
    m = d*r;
else
    m = d;
end

% Type of prior on dynamic parameters. Choices include matrix normal
% inverse Wishart on (A,Sigma) and normal on mu ('MNIW-N'), matrix normal
% inverse Wishart on (A,Sigma) with mean forced to 0 ('MNIW'), normal on A,
% inverse Wishart on Sigma, and normal on mu ('N-IW-N'), and fixed A,
% inverse Wishart on Sigma, and normal on mu ('Afixed-IW-N').  NOTE: right
% now, the 'N-IW-N' option is only coded for shared A!!!
model.obsModel.priorType = priorType;

switch model.obsModel.priorType
    case 'NIW'

        model.obsModel.params.M  = zeros([d 1]);
        model.obsModel.params.K =  kappa;
        
    case 'IW-N'
        % Mean and covariance for Gaussian prior on mean:
        model.obsModel.params.mu0 = zeros(d,1);
        model.obsModel.params.cholSigma0 = chol(sig0*eye(d));
    
    case 'MNIW'
        % Mean and covariance for A matrix:
        model.obsModel.params.M  = zeros([d m]);

        % Inverse covariance along rows of A (sampled Sigma acts as
        % covariance along columns):
        model.obsModel.params.K =  K(1:m,1:m);
        
    case 'MNIW-N'
        % Mean and covariance for A matrix:
        model.obsModel.params.M  = zeros([d m]);

        % Inverse covariance along rows of A (sampled Sigma acts as
        % covariance along columns):
        model.obsModel.params.K =  K(1:m,1:m);

        % Mean and covariance for mean of process noise:
        model.obsModel.params.mu0 = zeros(d,1);
        model.obsModel.params.cholSigma0 = chol(sig0*eye(d));

    case 'N-IW-N'
        % Mean and covariance for A matrix:
        model.obsModel.params.M  = zeros([d m]);
        model.obsModel.params.Lambda0_A = inv(kron(inv(K),meanSigma));

        % Mean and covariance for mean of process noise:
        model.obsModel.params.mu0 = zeros(d,1);
        model.obsModel.params.cholSigma0 = chol(sig0*eye(d));
        
    case 'Afixed-IW-N'
        % Set fixed A matrix:
        model.obsModel.params.A = A_shared;
        
        % Mean and covariance for mean of process noise:
        model.obsModel.params.mu0 = zeros(d,1);
        model.obsModel.params.cholSigma0 = chol(sig0*eye(d));
        
    case 'ARD'
        % Gamma hyperprior parameters for prior on precision parameter:
        model.obsModel.params.a_ARD = 10;
        model.obsModel.params.b_ARD = 0.01;
        
        % Placeholder for initializeStructs. Can I get rid of this?
        model.obsModel.params.M  = zeros([d m]);

        % Mean and covariance for mean of process noise:
        model.obsModel.params.zeroMean = 1;
end
        
% Degrees of freedom and scale matrix for covariance of process noise:
model.obsModel.params.nu = 1000; %d + 2;
model.obsModel.params.nu_delta = (model.obsModel.params.nu-d-1)*meanSigma;

if strcmp(obsModelType,'SLDS')
    % Degrees of freedom and scale matrix for covariance of measurement noise:
    model.obsModel.y_params.nu = 1000; %dy + 2;
    model.obsModel.y_params.nu_delta = (model.obsModel.y_params.nu-dy-1)*y_var*eye(dy);
    
    model.obsModel.y_priorType = y_priorType;
    
    switch model.obsModel.y_priorType
        case 'NIW'
            
            model.obsModel.y_params.M  = zeros([dy 1]);
            model.obsModel.y_params.K =  kappa_y;
            
        case 'IW-N'
            % Mean and covariance for Gaussian prior on mean:
            model.obsModel.y_params.mu0 = zeros(dy,1);
            model.obsModel.y_params.cholSigma0 = chol(sig0_y*eye(dy));
    end
    
    % Fixed measurement matrix:
    model.obsModel.params.C = [eye(dy) zeros(dy,d-dy)];
    
    % Initial state covariance:
    model.obsModel.params.P0 = P0*eye(d);
end

% Always using DP mixtures emissions, with single Gaussian forced by
% Ks=1...Need to fix.
model.obsModel.mixtureType = 'infinite';

% Sticky HDP-HMM parameter settings:
model.HMMmodel.params.a_alpha=1;  % affects \pi_z
model.HMMmodel.params.b_alpha=0.01;
model.HMMmodel.params.a_gamma=1;  % global expected # of HMM states (affects \beta)
model.HMMmodel.params.b_gamma=0.01;
if settings.Ks>1
    model.HMMmodel.params.a_sigma = 1;
    model.HMMmodel.params.b_sigma = 0.01;
end
if isfield(settings,'Kr')
    if settings.Kr > 1
        model.HMMmodel.params.a_eta = 1;
        model.HMMmodel.params.b_eta = 0.01;
    end
end
model.HMMmodel.params.c=100;  % self trans
model.HMMmodel.params.d=1;
model.HMMmodel.type = 'HDP';


%%

% Generate data from the prior:
data_struct = generateData(model,settings,T);
while length(unique(data_struct.true_labels))==1
    data_struct = generateData(model,settings,T);
end

close all;
figure; plot(data_struct.obs'); hold on; plot(data_struct.true_labels,'r','LineWidth',2); hold off;
title('True State and Mode Sequences')


%%
for seq=1
    data_struct(1).test_cases = seq;
    for t=trial_vec

        settings.trial = t;  % Defines trial number, which is part of the filename used when saving stats

        HDPHMMDPinference(data_struct,model,settings)
    end
end