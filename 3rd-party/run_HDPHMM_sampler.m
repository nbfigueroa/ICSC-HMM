function [ChainStats] = run_HDPHMM_sampler(data_struct, options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Define Sampler Settings and Models for HDP-HMM %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pre-defined for Gaussian emission models with NIW prior

% Sampler Settings for HDP-HMM:
settings.Kz = options.Kz;              % truncation level for mode transition distributions
settings.Ks = options.Ks;              % truncation level for mode transition distributions
settings.Niter = options.Niter;         % Number of iterations of the Gibbs sampler
settings.resample_kappa = options.sticky;   % Whether or not to use sticky model
settings.seqSampleEvery = 100; % How often to run sequential z sampling
settings.saveEvery = 100;      % How often to save Gibbs sample stats
settings.storeEvery = 1;
settings.storeStateSeqEvery = 100;
settings.ploton = options.plot_iter;    % Plot the mode sequence while running sampler
settings.plotEvery = 20;
settings.plotpause = 0;        % Length of time to pause on the plot
settings.saveDir = options.saveDir;    % Directory to which to save files
settings.formZinit = 1;

% Dynamical Model Settings for HDP-HMM:
model.obsModel.type = options.obsModelType;
model.obsModel.priorType = options.priorType;
model.obsModel.params.M  = zeros([options.d 1]);
model.obsModel.params.K =  options.kappa;

% Degrees of freedom and scale matrix for covariance of process noise:
model.obsModel.params.nu = options.d + 2; %d + 2;
model.obsModel.params.nu_delta = (model.obsModel.params.nu-options.d-1)*options.meanSigma;
      
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

data_struct(1).test_cases = [1:length(data_struct)];
settings.trial = 1;
fprintf('Running HDP-HMM Gibbs Sampler... \n');
tic;
[ChainStats] = HDPHMMDPinference(data_struct,model,settings);
toc;


end