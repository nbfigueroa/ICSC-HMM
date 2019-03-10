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

%% 2a) Toy 2D dataset, 4 Unique Emission models, 5 time-series
clc; clear all; close all;
[data, TruePsi, Data, True_states] = genToySeqData_Gaussian( 4, 2, 2, 500, 0.5 ); 
dataset_name = '2D';

% Feat matrix F (binary 5 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

%% 2b) Toy 2D dataset, 2 Unique Emission models transformed, 4 time-series
clc; clear all; close all;
[data, TruePsi, Data, True_states] = genToySeqData_TR_Gaussian(4, 2, 3, 500, 0.5 );
dataset_name = '2D Transformed'; 

% Feat matrix F (binary 4 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

% Similarity matrix S (4 x 4 matrix)
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSimMat( TruePsi.S );

%% 3) Real 'Grating' 7D dataset, 3 Unique Emission models, 12 time-series
%Demonstration of a Carrot Grating Task consisting of 
%12 (7-d) time-series X = {x_1,..,x_T} with variable length T. 
%Dimensions:
% x = {pos_x, pos_y, pos_z, q_i, q_j, q_k, q_w}
% type= 'robot'/'grater'/'mixed'
clc; clear all; close all;
data_path = './test-data/'; display = 1; type = 'mixed'; full = 0; use_vel = 0;
% Type of data processing
% O: no data manipulation -- 1: zero-mean -- 2: scaled by range * weights
normalize = 1; 
[data, TruePsi, Data, True_states ,Data_] = load_grating_dataset( data_path, type, display, full, normalize, use_vel);
dataset_name = 'Grating'; 

%% 4) Real 'Dough-Rolling' 12D dataset, 3 Unique Emission models, 12 time-series
% Demonstration of a Dough Rolling Task consisting of 
% 15 (13-d) time-series X = {x_1,..,x_T} with variable length T. 
%
% Dimensions:
% x = {pos_x, pos_y, pos_z, q_i, q_j, q_k, q_w, f_x, f_y, f_z, tau_x, tau_y, tau_z}
% - positions:         Data{i}(1:3,:)   (3-d: x, y, z)
% - orientations:      Data{i}(4:7,:)   (4-d: q_i, q_j, q_k, q_w)
% - forces:            Data{i}(8:10,:)   (3-d: f_x, f_y, f_z)
% - torques:           Data{i}(11:13,:) (3-d: tau_x, tau_y, tau_z)

% Dataset type:
%
% type: 'raw', raw sensor recordings at 500 Hz, f/t readings are noisy af and
% quaternions dimensions exhibit discontinuities
% This dataset is NOT labeled
%
% type: 'proc', sub-sampled to 100 Hz, smoothed f/t trajectories, fixed rotation
% discontinuities.

clc; clear all; close all;
data_path = './test-data/'; display = 1; type = 'proc'; full = 0; type2 = 'real'; 
% Type of data processing
% O: no data manipulation -- 1: zero-mean -- 2: scaled by range * weights
normalize = 2; 

% Define weights for dimensionality scaling
weights = [10*ones(1,3) 2*ones(1,4) 1/7*ones(1,3) 1/10*ones(1,3)]';

% Define if using first derivative of pos/orient
use_vel = 1;
[data, TruePsi, Data, True_states, Data_] = load_rolling_dataset( data_path, type, type2, display, full, normalize, weights, use_vel);
dataset_name = 'Rolling'; 


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
weights = [3*ones(1,3) 1/2*ones(1,4) 1/15*ones(1,3) 1/2*ones(1,3)]';
switch dim                
    case 'active'
    case 'robots' 
        weights = [weights 1/3*ones(1,3) 2*ones(1,4) 1/15*ones(1,3) 1/5*ones(1,3)]';        
end

% Define if using first derivative of pos/orient
use_vel = 0;

[data, TruePsi, Data, True_states, Data_] = load_peeling_dataset( data_path, dim, display, normalize, weights, use_vel);
dataset_name = 'Peeling';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Run Collapsed ICSC-HMM Sampler T times for good statistics          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define Settings for IBP-HMM %%%

% Initialize parallel computing
% parpool;

% IBP initial hyper-parametrs
gamma = length(Data); 
alpha = 1;  
kappa = 20; 

% Model Setting (IBP mass, IBP concentration, HMM alpha, HMM sticky)
modelP = {'bpM.gamma', gamma, 'bpM.c', 1, 'hmmM.alpha', alpha, 'hmmM.kappa', kappa}; 

% Sampler Settings
algP   = {'Niter', 500, 'HMM.doSampleHypers', 1,'BP.doSampleMass', 1, 'BP.doSampleConc', 0, ...
         'doSampleFUnique', 1, 'doSplitMerge', 0}; 

% Number of Repetitions
T = 3; 
% Run MCMC Sampler for T times
Sampler_Stats = [];
jobID = ceil(rand*1000);
for run=1:T       
    % Run MCMC Sampler for Niter once.
    clear CH    
    initP  = {'F.nTotal', randsample(data.N,1)+1}; 
    CH = runICSCHMM( data, modelP, {jobID, run}, algP, initP, './icsc-Results');  
    Sampler_Stats(run).CH = CH;
end

%% %%%%%%%% Visualize Sampler Convergence/Metrics and extract Best Psi/run %%%%%%%%%%

%%%%%% Compute Clustering/Segmentation Metrics vs Ground Truth %%%%%%
if isfield(TruePsi, 'sTrueAll')
    true_states_all = TruePsi.sTrueAll;
else
    true_states_all = data.zTrueAll;
end

if exist('h1','var')  && isvalid(h1),  delete(h1);end
if exist('h1b','var') && isvalid(h1b), delete(h1b);end
[h1, h1b, Best_Psi] = plotSamplerStatsBestPsi(Sampler_Stats,'metrics', true_states_all);

% Compute metrics for ICSC-HMM
clc;
results = computeSegmClustmetrics(true_states_all, Best_Psi);

%% Choose best run
log_probs = zeros(1,T);
for ii=1:T; log_probs(ii) = Best_Psi(ii).logPr; end

[val_max id_max] = sort(log_probs,'descend')

%% Plot Segmentation+Clustering with Chosen Run and Metrics

% Choose best IBP-HMM run
bestPsi = Best_Psi(id_max(1));
est_labels = bestPsi.Psi.Z;

if exist('h2','var') && isvalid(h2), delete(h2);end
[ h2 ] = plotDoubleLabelSegmentation(data, bestPsi);

% Plot Estimated Feature Matrix
if exist('h3','var') && isvalid(h3), delete(h3);end
[ h3 ] = plotFeatMat( bestPsi.Psi.F);

% Plot Estimated Transition Matrices
if exist('h4','var') && isvalid(h4), delete(h4);end
[h4, bestPsi] = plotTransitionMatrices(bestPsi);

% Compute Segmentation and State Clustering Metrics
results = computeSegmClustmetrics(true_states_all, bestPsi);

%% Visualize Estimated  Emission Parameters for 2D Datasets ONLY!
title_name  = 'Estimated Emission Parameters';
plot_labels = {'$x_1$','$x_2$'};
clear Est_theta
Est_theta.K = bestPsi.nFeats;
for k=1:Est_theta.K
    Est_theta.Mu(:,k)         = bestPsi.Psi.theta(k).mu;
    Est_theta.invSigma(:,:,k) = bestPsi.Psi.theta(k).invSigma;
    Est_theta.Sigma(:,:,k)    = Est_theta.invSigma(:,:,k) \ eye(data.D);
end

if exist('h4','var') && isvalid(h4), delete(h4);end
h4 = plotGaussianEmissions2D(Est_theta, plot_labels, title_name, est_labels);

%% Visualize Segmented Trajectories in 3D ONLY!
labels = [];
labels_c = [];
for e=1:length(bestPsi.Psi.stateSeq)
    est_states{e} = bestPsi.Psi.stateSeq(e).z';
    est_clusts{e} = bestPsi.Psi.stateSeq(e).c';
    labels = [labels unique(est_states{e})'];    
    labels_c = [labels_c unique(est_clusts{e})'];    
end
labels = unique(labels);
labels_c = unique(labels_c);

O = eye(4); O(1,4) = -0.3;O(2,4) = -0.5;
% Plot Segmentated 3D Trajectories
titlename = strcat(dataset_name,' Demonstrations (Transform-Dependent Segmentation)');
if exist('h5','var') && isvalid(h5), delete(h5);end
h5 = plotLabeled3DTrajectories(Data_, est_states, titlename, labels);
drawframe(O, 0.05); 
axis tight

% Plot Clustered/Segmentated 3D Trajectories
titlename = strcat(dataset_name,' Demonstrations (Transform-Invariant Segmentation)');
if exist('h6','var') && isvalid(h6), delete(h6);end
h6 = plotLabeled3DTrajectories(Data_, est_clusts, titlename, labels_c);
drawframe(O, 0.05); 
axis tight

%% Plot Segmentated 3D Trajectories
titlename = strcat(dataset_name,' Demonstrations (Ground Truth)');
if exist('h7','var') && isvalid(h7), delete(h7);end
h7 = plotLabeled3DTrajectories(Data_, True_states, titlename, unique(data.zTrueAll));
drawframe(O, 0.05); 
axis tight
