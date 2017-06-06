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
label_range = unique(True_states{1});

%% 2a) Toy 2D dataset, 4 Unique Emission models, 5 time-series
clc; clear all; close all;
[data, TruePsi, Data, True_states] = genToySeqData_Gaussian( 4, 2, 2, 500, 0.5 ); 
dataset_name = '2D'; Data_ = Data;
label_range = unique(data.zTrueAll);

% Feat matrix F (binary 5 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

%% 2b) Toy 2D dataset, 2 Unique Emission models transformed, 5 time-series
clc; clear all; close all;
[data, TruePsi, Data, True_states] = genToySeqData_TR_Gaussian(4, 2, 4, 500, 0.5 );
dataset_name = '2D Transformed'; Data_ = Data;
label_range = unique(data.zTrueAll);

% Feat matrix F (binary 4 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

%% 3) Real 'Grating' 7D dataset, 3 Unique Emission models, 12 time-series
%Demonstration of a Carrot Grating Task consisting of 
%12 (7-d) time-series X = {x_1,..,x_T} with variable length T. 
%Dimensions:
%x = {pos_x, pos_y, pos_z, q_i, q_j, q_k, q_w}
% clc; clear all; close all;
data_path = './test-data/'; display = 1; type = 'same'; full = 0;
[data, ~, Data, True_states] = load_grating_dataset( data_path, type, display, full);
dataset_name = 'Grating'; Data_ = Data;

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
data_path = './test-data/'; display = 1; type = 'proc'; full = 0; 
% Type of data processing
% O: no data manipulation -- 1: zero-mean -- 2: scaled by range * weights
normalize = 2; 

% Define weights for dimensionality scaling
weights = [5*ones(1,3) ones(1,4) 1/10*ones(1,3) 0*ones(1,3)]';

% Define if using first derivative of pos/orient
use_vel = 1;
[data, TruePsi, ~, True_states, Data_] = load_rolling_dataset( data_path, type, display, full, normalize, weights, use_vel);
dataset_name = 'Rolling';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Run Collapsed IBP-HMM Sampler T times for good statistics          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define Settings for IBP-HMM %%%

% IBP hyper-parametrs
gamma = length(Data_);  % length(Data)
alpha = 1;  % typically 1.. could change
kappa = 10; % sticky parameter

% Model Setting (IBP mass, IBP concentration, HMM alpha, HMM sticky)
modelP = {'bpM.gamma', gamma, 'bpM.c', 1, 'hmmM.alpha', alpha, 'hmmM.kappa', kappa}; 

% Sampler Settings
algP   = {'Niter', 500, 'HMM.doSampleHypers',0,'BP.doSampleMass',1,'BP.doSampleConc', 1, ...
         'doSampleFUnique', 1. 'doSplitMerge', 0}; 

% Number of Repetitions
T = 5; 

% Run MCMC Sampler for T times
Sampler_Stats = [];
jobID = ceil(rand*1000);
for run=1:T       
    % Run Gibbs Sampler for Niter once.
    clear CH    
    % Start out with just one feature for all objects
    initP  = {'F.nTotal', randsample(ceil(data.N),1)}; 
    CH = runBPHMM( data, modelP, {jobID, run}, algP, initP, './ibp-Results' );  
    Sampler_Stats(run).CH = CH;
end

%%%%%%%%%% Visualize Sampler Convergence and Best Psi/run %%%%%%%%%%
if exist('h1','var') && isvalid(h1), delete(h1);end
[h1, Best_Psi] = plotSamplerStatsBestPsi(Sampler_Stats);

%%%%%% Compute Clustering/Segmentation Metrics vs Ground Truth %%%%%%
% Segmentation Metric Arrays
hamming_distance   = zeros(1,T);
global_consistency = zeros(1,T);
variation_info     = zeros(1,T);
inferred_states    = zeros(1,T);

% Clustering Metric Arrays
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);

true_states_all = data.zTrueAll;

for run=1:T    
    clear Psi
    est_states_all = [];
    
    % Extract Estimated States for all sequences
    Psi = Best_Psi(run).Psi;
    for j=1:data.N
        est_states_all = [est_states_all Best_Psi(run).Psi.stateSeq(j).z];
    end
    
     % Segmentation Metrics per run
    [relabeled_est_states_all, hamming_distance(run),~,~] = mapSequence2Truth(true_states_all,est_states_all);
    [~,global_consistency(run), variation_info(run)] = compare_segmentations(true_states_all,est_states_all);
    inferred_states(run)   = length(unique(est_states_all));
    
    % Cluster Metrics per run
    [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_states_all, relabeled_est_states_all);
    
end

% Overall Stats for HMM segmentation and state clustering
fprintf('*** IBP-HMM Results*** \n Optimal States: %3.3f (%3.3f) \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) NMI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n',[mean(inferred_states) std(inferred_states) mean(hamming_distance) std(hamming_distance)  ...
    mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])


%% Visualize Transition Matrices, Emission Parameters and Segmentation from 'Best' Run

% Get 'Best Psi' from all runs
log_probs = zeros(1,T);
for ii=1:T 
    mean_likelihoods(ii) = mean(Best_Psi(ii).logPr); 
    std_likelihoods(ii) = std(Best_Psi(ii).logPr); 
end
% [Max_ll, max_id] = max(log_likelihoods)
[val_std id_std] = sort(std_likelihoods,'ascend');
[val_mean id_mean] = sort(mean_likelihoods,'descend');

id_mean
id_std

%% Plot Segmentation with Chosen Run
id = 5;
bestPsi = Best_Psi(id);

% Extract info from 'Best Psi'
K_est = bestPsi.nFeats;
est_states_all = [];
for ii=1:data.N; est_states_all  = [est_states_all bestPsi.Psi.stateSeq(ii).z]; end
label_range = unique(est_states_all);
est_states = [];

% Plot Segmentation
figure('Color',[1 1 1])
true_states_all   = [];
est_states_all    = [];

for i=1:data.N
    
    % Extract data from each time-series    
    X = data.Xdata(:,[data.aggTs(i)+1:data.aggTs(i+1)]);
    
    % Segmentation Direct from state sequence (Gives the same output as Viterbi estimate)
    est_states{i}  = bestPsi.Psi.stateSeq(i).z;
    
    % Stack labels for state clustering metrics
    true_states_all = [true_states_all; True_states{i}];
    est_states_all  = [est_states_all; est_states{i}'];
    
    % Plot Inferred Segments
    subplot(data.N,1,i);
    data_labeled = [X; est_states{i}];
    plotLabeledData( data_labeled, [], strcat('Segmented Time-Series (', num2str(i),'), K:',num2str(K_est)), [],label_range)
    
end

% Segmentation Metrics per run
[relabeled_est_states_all, hamming_distance_,~,~] = mapSequence2Truth(true_states_all,est_states_all);
[~,global_consistency_, variation_info_] = compare_segmentations(true_states_all,est_states_all);
inferred_states_   = length(unique(est_states_all));

% Cluster Metrics per run
[cluster_purity_ cluster_NMI_ cluster_F_] = cluster_metrics(true_states_all, relabeled_est_states_all);

% Overall Stats for HMM segmentation and state clustering
fprintf('*** IBP-HMM Results*** \n Optimal States: %3.3f \n Hamming-Distance: %3.3f GCE: %3.3f VO: %3.3f \n Purity: %3.3f  NMI: %3.3f  F: %3.3f   \n',[inferred_states_  hamming_distance_  ...
    global_consistency_ variation_info_ cluster_purity_ cluster_NMI_ cluster_F_])


% Plot Estimated Feature Matrix
if exist('h2','var') && isvalid(h2), delete(h2);end
h2 = plotFeatMat( bestPsi.Psi.F);

% Plot Estimated Transition Matrices
if exist('h3','var') && isvalid(h3), delete(h3);end
h3 = figure('Color',[1 1 1]);
pi = [];
for i=1:data.N   
    % Construct Transition Matrices for each time-series
    f_i   = bestPsi.Psi.Eta(i).availFeatIDs;
    eta_i = bestPsi.Psi.Eta(i).eta;
    
    % Normalize self-transitions with sticky parameter    
    pi_i  = zeros(size(eta_i));   
    for ii=1:size(pi_i,1);pi_i(ii,:) = eta_i(ii,:)/sum(eta_i(ii,:));end
    pi{i} = pi_i;    
    
    % Plot them
    subplot(data.N, 1, i)
    plotTransMatrix(pi{i},strcat('Time-Series (', num2str(i),')'),0, f_i);
end
bestPsi.pi = pi;

%% Plot Estimated  Emission Parameters for 2D Datasets ONLY!
title_name  = 'Estimated Emission Parameters';
plot_labels = {'$x_1$','$x_2$'};
clear Est_theta
Est_theta.K = K_est;
for k=1:K_est
    Est_theta.Mu(:,k)         = bestPsi.Psi.theta(k).mu;
    Est_theta.invSigma(:,:,k) = bestPsi.Psi.theta(k).invSigma;
    Est_theta.Sigma(:,:,k)    = Est_theta.invSigma(:,:,k) \ eye(data.D);
end

if exist('h4','var') && isvalid(h4), delete(h4);end
h4 = plotGaussianEmissions2D(Est_theta, plot_labels, title_name, label_range);

%% Visualize Segmented Trajectories in 3D ONLY!
labels    = unique(est_states_all);
titlename = strcat(dataset_name,' Demonstrations (Estimated Segmentation)');

% Plot Segmentated 3D Trajectories
if exist('h5','var') && isvalid(h5), delete(h5);end
h5 = plotLabeled3DTrajectories(Data_, est_states, titlename, labels);

titlename = strcat(dataset_name,' Demonstrations (Ground Truth)');
% Plot Segmentated 3D Trajectories
if exist('h6','var') && isvalid(h6), delete(h6);end
h6 = plotLabeled3DTrajectories(Data_, True_states, titlename, unique(data.zTrueAll));

