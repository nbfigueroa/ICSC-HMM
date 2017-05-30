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
[data, ~, True_states] = genToyHMMData_Gaussian( N_TS, display ); 
label_range = unique(True_states{1});

%% 2) Toy 2D dataset, 4 Unique Emission models, 5 time-series
clc; clear all; close all;
[data, TruePsi, ~, True_states] = genToySeqData_Gaussian( 4, 2, 5, 500, 0.5 ); 
label_range = unique(data.zTrueAll);

% Feat matrix F (binary 5 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Run Sticky HDP-HMM Sampler T times for good statistics             %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define Settings for IBP-HMM %%%

% Model Setting (IBP mass, IBP concentration, HMM alpha, HMM sticky)
modelP = {'bpM.gamma', 1, 'bpM.c', 1, 'hmmM.alpha', 1, 'hmmM.kappa', 10}; 

% Sampler Settings
algP   = {'Niter', 500, 'HMM.doSampleHypers',0,'BP.doSampleMass',1,'BP.doSampleConc',1}; 

% Number of Repetitions
T = 10; 

% Segmentation Metric Arrays
hamming_distance   = zeros(1,T);
global_consistency = zeros(1,T);
variation_info     = zeros(1,T);
inferred_states    = zeros(1,T);

% Clustering Metric Arrays
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);

Sampler_Stats = [];
jobID = ceil(rand*1000);
for run=1:T       
    % Run Gibbs Sampler for Niter once.
    clear CH    
    % Start out with just one feature for all objects
    initP  = {'F.nTotal', randsample(T,1)}; 
    CH = runBPHMM( data, modelP, {jobID, run}, algP, initP, './IBP-Results' );  
    Sampler_Stats(run).CH = CH;
end

%% %%%%%% Visualize MCMC Sampler Convergence %%%%%%%%%%%%%%

% Gather High-level Stats
T = length(Sampler_Stats);
Iterations = Sampler_Stats(1).CH.iters.logPr;

% Plot Joint Log-prob
figure('Color',[1 1 1])
subplot(2,1,1)
for i=1:T
    joint_logs = zeros(1,length(Iterations));
    for ii=1:length(Iterations); joint_logs(1,ii) = Sampler_Stats(i).CH.logPr(ii).all;end
    [max_joint best_iter] = max(joint_logs);
    semilogx(Iterations,joint_logs,'--*', 'LineWidth', 2,'Color',[rand rand rand]); hold on;
end
xlim([1 Iterations(end)])
xlabel('MCMC Iteration','Interpreter','LaTex','Fontsize',20); ylabel('LogPr','Interpreter','LaTex','Fontsize',20)
title ({sprintf('Trace of Joint Probabilities $p(F, S, X)$ for %d runs',[T])}, 'Interpreter','LaTex','Fontsize',20)
grid on

Iterations_feat = Sampler_Stats(1).CH.iters.Psi;
subplot(2,1,2)
for i=1:T
    nFeats = zeros(1,length(Iterations_feat));
    for ii=1:length(Iterations_feat); nFeats(1,ii) = length(Sampler_Stats(i).CH.Psi(ii).theta);end
    
    stairs(Iterations_feat,nFeats, 'LineWidth',2); hold on;
    set(gca, 'XScale', 'log')
    xlim([1 Iterations_feat(end)])
end
xlim([1 Iterations_feat(end)])
xlabel('MCMC Iteration','Interpreter','LaTex','Fontsize',20); ylabel('$K$','Interpreter','LaTex','Fontsize',20)
title ({sprintf('Number of estimated features (shared states) for %d runs',[T])}, 'Interpreter','LaTex','Fontsize',20)
grid on

%% %%%%%% Visualize MCMC Sampler Clustering/Segmentation vs Ground Truth %%%%%%%%%%%%%%


%% VISUALIZE RESULTS!
