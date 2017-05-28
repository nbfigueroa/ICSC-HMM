%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main demo scripts for the ICSC-HMM Segmentation Algorithm proposed in:
%
% N. Figueroa and A. Billard, “Transform-Invariant Clustering of SPD Matrices 
% and its Application on Joint Segmentation and Action Discovery}”
% Arxiv, 2017. 
%
% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% November 2016; Last revision: 25-May-2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    --Select a Dataset to Test--                       %%    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1) Toy 2D dataset, 3 Unique Emission models, 3 time-series, same swicthing
clc; clear all; close all;
N_TS = 3; display = 2 ; % 0: no-display, 1: raw data in one plot, 2: ts w/labels
[Data, True_states] = genToyHMMData_Gaussian( N_TS, display ); 
label_range = unique(True_states{1});

%% 2) Toy 2D dataset, 4 Unique Emission models, 5 time-series
clc; clear all; close all;
[data, TruePsi, Data, True_states] = genToySeqData_Gaussian( 4, 2, 5, 500, 0.5 ); 
label_range = unique(data.zTrueAll);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Run Sticky HDP-HMM Sampler N times for good statistics             %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Default Setting for HDP-HMM
obsModelType = 'Gaussian';
priorType = 'NIW';
d = size(Data{1},2);
kappa = 0.1;           % NIW(kappa,theta,delta,nu_delta)
meanSigma = eye(d);    % expected mean of IW(nu,nu_delta) prior on Sigma_{k,j}
Kz = 10;               % truncation level of the DP prior on HMM transition distributions pi_k
Ks = 1;                % truncation level of the DPMM on emission distributions pi_s
saveDir = './Results';

% Sampler Settings for HDP-HMM:
clear settings
settings.Kz = Kz;              % truncation level for mode transition distributions
settings.Ks = Ks;              % truncation level for mode transition distributions
settings.Niter = 2000;         % Number of iterations of the Gibbs sampler
settings.resample_kappa = 1;   % Whether or not to use sticky model
settings.seqSampleEvery = 100; % How often to run sequential z sampling
settings.saveEvery = 100;      % How often to save Gibbs sample stats
settings.storeEvery = 1;
settings.storeStateSeqEvery = 100;
settings.ploton = 1;           % Plot the mode sequence while running sampler
settings.plotEvery = 20;
settings.plotpause = 0;        % Length of time to pause on the plot
settings.saveDir = saveDir;    % Directory to which to save files

% Dynamical Model Settings for HDP-HMM:
clear model
model.obsModel.type = obsModelType;
model.obsModel.priorType = priorType;
model.obsModel.params.M  = zeros([d 1]);
model.obsModel.params.K =  kappa;
% Degrees of freedom and scale matrix for covariance of process noise:
model.obsModel.params.nu = 1000; %d + 2;
model.obsModel.params.nu_delta = (model.obsModel.params.nu-d-1)*meanSigma;
      
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

