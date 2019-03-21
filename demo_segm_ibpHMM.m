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
[data, Data, True_states, True_theta] = genToyHMMData_Gaussian( N_TS, display ); 
super_states = 0;

%% 2a) Toy 2D dataset, 4 Unique Emission models, 5 time-series
clc; clear all; close all;
[data, TruePsi, Data, True_states] = genToySeqData_Gaussian( 4, 2, 2, 500, 0.5 ); 
dataset_name = '2D';
super_states = 0;

% Feat matrix F (binary 5 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

%% 2b) Toy 2D dataset, 2 Unique Emission models transformed, 4 time-series
clc; clear all; close all;
[data, TruePsi, Data, True_states] = genToySeqData_TR_Gaussian(4, 2, 3, 500, 0.5 );
dataset_name = '2D Transformed'; 
super_states = 1;

% Feat matrix F (binary 4 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

% Similarity matrix S (4 x 4 matrix)
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSimMat( TruePsi.S );

%% 5) Real 'Peeling' (max) 32-D dataset, 5 Unique Emission models, 3 time-series
% Demonstration of a Bimanual Peeling Task consisting of 
% 3 (32-d) time-series X = {x_1,..,x_T} with variable length T. 
% Dimensions:
% x_a = {pos_x, pos_y, pos_z, q_i, q_j, q_k, q_w, f_x, f_y, f_z, tau_x, tau_y, tau_z}
% - positions:              Data{i}(1:3,:)   (3-d: x, y, z)
% - orientations:           Data{i}(4:7,:)   (4-d: q_i, q_j, q_k, q_w)
% - forces:                 Data{i}(8:10,:)  (3-d: f_x, f_y, f_z)
% - torques:                Data{i}(11:13,:) (3-d: tau_x, tau_y, tau_z)
% x_p = {pos_x, pos_y, pos_z, q_i, q_j, q_k, q_w, f_x, f_y, f_z, tau_x, tau_y, tau_z}
% - same as above           Data{i}(14:26,:)
% x_o = {mu_r, mu_g, mu_b, sigma_r, sigma_g, sigma_b}
% - rate_mean:              Data{i}(27:29,:)   (3-d: mu_r, mu_g, mu_b)
% - rate_variance:          Data{i}(30:32,:)   (3-d: sigma_r, sigma_g, sigma_b)

% Dimension type:
% dim: 'all', include all 32 dimensions (active + passive robots + object)
% dim: 'robots', include only 26-d from measurements from active + passive robots
% dim: 'act+obj', include only 19-d from measurements from active robot + object
% dim: 'active', include only 13-d from measurements from active robot

% Dataset type:
% sub-sampled to 100 Hz (from 500 Hz), smoothed f/t trajectories, fixed rotation
% discontinuities.

clc; 
clear all; close all
data_path = './test-data/'; display = 1; 

% Type of data processing
% O: no data manipulation -- 1: zero-mean -- 2: scaled by range * weights
normalize = 2; 

% Select dimensions to use
dim = 'active'; 

% Define weights for dimensionality scaling
weights = [3*ones(1,3) 1/2*ones(1,4) 1/10*ones(1,3) 1/20*ones(1,3)]';
switch dim                
    case 'active'
    case 'robots' 
        weights = [weights' 1/7*ones(1,3) 1/2*ones(1,4) 1/20*ones(1,3) 1/10*ones(1,3)]';        
end


% Select dimensions to use
dim = 'robots'; 

% Define weights for dimensionality scaling
switch dim
    case 'active'
        weights = [5*ones(1,3) 2*ones(1,4) 1/10*ones(1,6)]'; % active    
    case 'act+obj'
        weights = [2*ones(1,7) 1/10*ones(1,6) 2*ones(1,6)]'; % act+obj
    case 'robots'
%         weights = [5*ones(1,3) ones(1,4) 1/10*ones(1,6) ones(1,3) 2*ones(1,4) 1/20*ones(1,6) ]'; % robots(velocities)        
        weights = [5*ones(1,3) 1/2*ones(1,4) 1/10*ones(1,6) ones(1,3) 1/2*ones(1,4) 1/20*ones(1,6) ]'; % robots(position)        
    case 'all'
        weights = [5*ones(1,3) ones(1,4) 1/10*ones(1,6) ones(1,3) 2*ones(1,4) 1/20*ones(1,6) 2*ones(1,6) ]'; % all        
end


% Define if using first derivative of pos/orient
use_vel = 0;

[data, TruePsi, Data, True_states, Data_] = load_peeling_dataset( data_path, dim, display, normalize, weights, use_vel);
dataset_name = 'Peeling';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Step 2: Run Collapsed IBP-HMM Sampler T times for good statistics    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define Settings for IBP-HMM %%%
% IBP hyper-parametrs
gamma = 2*length(Data);  % length(Data)
alpha = 1;  % typically 1.. could change
kappa = 10; % sticky parameter

% Model Setting (IBP mass, IBP concentration, HMM alpha, HMM sticky)
modelP = {'bpM.gamma', gamma, 'bpM.c', 1, 'hmmM.alpha', alpha, 'hmmM.kappa', kappa,'obsM.Scoef',1}; 

% Sampler Settings
algP   = {'Niter', 1000, 'HMM.doSampleHypers', 1, 'BP.doSampleMass',1,'BP.doSampleConc', 0, ...
         'doSampleFUnique', 1, 'doSplitMerge', 0}; 

% Number of Repetitions
T = 10; 

% Run MCMC Sampler for T times
Sampler_Stats = [];
jobID = ceil(rand*1000);
for run=1:T       
    % Run Gibbs Sampler for Niter once.
    clear CH    
    % Start out with random number of features
    initP  = {'F.nTotal', randsample(10,1)}; 
    CH = runBPHMM( data, modelP, {jobID, run}, algP, initP, './ibp-Results' );  
    Sampler_Stats(run).CH = CH;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               Extract Estimation Statistics         %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Visualize Sampler Convergence/Metrics and extract Best Psi/run %%%%%%%%%%
if exist('h1','var')  && isvalid(h1),  delete(h1);end
if exist('h1b','var') && isvalid(h1b), delete(h1b);end
% [h1, h1b, Best_Psi] = plotSamplerStatsBestPsi(Sampler_Stats);
[h1, h1b, Best_Psi] = plotSamplerStatsBestPsi(Sampler_Stats, 'hist');

%%%%%% Compute Clustering/Segmentation Metrics vs Ground Truth %%%%%%
if isfield(TruePsi, 'sTrueAll')
    true_states_all = TruePsi.sTrueAll;
else
    true_states_all = data.zTrueAll;
end

% Compute metrics for IBP-HMM
ibp_stats = computeSegmClustmetrics(true_states_all, Best_Psi);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              Extract Best Run and Plot Results            %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Choose best run
log_probs = zeros(1,T);
for ii=1:T; log_probs(ii) = Best_Psi(ii).logPr; end
[val_max, id_max] = sort(log_probs,'descend');
bestPsi           = Best_Psi(id_max(1));

% Plot Segmentation with Chosen Run
if exist('h2','var') && isvalid(h2), delete(h2);end
title_name = strcat(dataset_name, ': $\mathcal{IBP}$-HMM Segmentation');
[ h2 ] = plotSingleLabelSegmentation(data, bestPsi, data_legends, title_name);

% Plot Estimated Feature Matrix
if exist('h3','var') && isvalid(h3), delete(h3);end
[ h3 ] = plotFeatMat( bestPsi.Psi.F);

% Plot Estimated Transition Matrices
if exist('h4','var') && isvalid(h4), delete(h4);end
[h4, bestPsi] = plotTransitionMatrices(bestPsi);

% Compute Segmentation and State Clustering Metrics
results = computeSegmClustmetrics(true_states_all, bestPsi);

% Plot Segmentated 3D Trajectories
est_states = [];
for e=1:length(bestPsi.Psi.stateSeq)
    est_states{e} = bestPsi.Psi.stateSeq(e).z';
end

N = size(Data{1},2);
if N > 2    
    % Plot Ground Segmentations
    if exist('h5','var') && isvalid(h5), delete(h5);end
    title_name = {dataset_name,'Ground Truth'};
    title_edge = [0 0 0];
    h5 = plot3D_labeledTrajectories(Data,True_states,plot_subsample,title_name, title_edge,  dataset_view);    
    
    % Plot Estimated Segmentations
    if exist('h6','var') && isvalid(h6), delete(h6);end
    title_name = {dataset_name,' $\mathcal{IBP}$-HMM Segmentation'};
    title_edge = [0.470588235294118 0.670588235294118 0.188235294117647];
    h6 = plot3D_labeledTrajectories(Data,est_states,plot_subsample,title_name, title_edge, dataset_view);
end
