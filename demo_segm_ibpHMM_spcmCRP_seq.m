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
% [data, TruePsi] = genToySeqData_TR_Gaussian(4, 2, 3, 500, 0.5 );
dataset_name = '2D Transformed'; 
super_states = 1;

% Feat matrix F (binary 4 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

% Similarity matrix S (4 x 4 matrix)
title_str = 'SPCM Similarity Matrix of Emission models';
h1 = plotSimilarityConfMatrix(TruePsi.S, title_str);
 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Step 1 (a): Load One of the Wiping Datasets      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc; clear all; close all;
data_path  = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/ICSC-HMM/test-data/';

%%%%%%%%%%%  Input Data loading function %%%%%%%%%%
% Select Dataset to Load
dataset_type = 2; % 1: Rim Cover, 2: Door Fender                   
% Select which Variables to use
variables  = 1;
do_plots   = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data, TruePsi, Data, True_states, ...
    data_legends, dataset_name, plot_subsample, dataset_view] = load_wiping_dataset(data_path, dataset_type, do_plots, variables);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Step 1 (b): Load Rolling Dataset      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc; clear all; close all;
data_path  = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/ICSC-HMM/test-data/';
%%%%%%%%%%%  Input Data loading function %%%%%%%%%%
variables  = 1;
do_plots   = 1;
time_series = 5; % Max is 15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data, TruePsi, Data, True_states, ...
    data_legends, dataset_name, plot_subsample, dataset_view] = load_rolling_dataset2(data_path, time_series, do_plots, variables);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Step 1 (c): Load Peeling Dataset      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real 'Peeling' (max) 32-D dataset, 5 Unique Emission models, 3 time-series
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

% clc;  clear all; 
% close all
data_path  = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/ICSC-HMM/test-data/';

%%%%%%%%%%%  Input Data loading function %%%%%%%%%%
do_plots = 1; 
dim = 'robots'; % used 'active' or 'act+obj'
time_series = 3; % Max is 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data, TruePsi, Data, True_states, ...
    data_legends, dataset_name, plot_subsample, dataset_view] = load_peeling_dataset2( data_path, do_plots, dim, time_series);

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
    initP  = {'F.nTotal', randsample(10,1)+1}; 
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
ibp_results = computeSegmClustmetrics(true_states_all, bestPsi);

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
    h5 = plot3D_labeledTrajectories(Data,True_states,plot_subsample, title_name, title_edge,  dataset_view, dim);    
    
    % Plot Estimated Segmentations
    if exist('h6','var') && isvalid(h6), delete(h6);end
    title_name = {dataset_name,' $\mathcal{IBP}$-HMM Segmentation'};
    title_edge = [0.470588235294118 0.670588235294118 0.188235294117647];
    h6 = plot3D_labeledTrajectories(Data,est_states,plot_subsample, title_name, title_edge, dataset_view, dim);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       Step 3: Create Vector Space Embeddings of HMM Emission Models %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract Emission Parameters
labels = [];
for e=1:length(est_states)
    labels = [labels unique(est_states{e})'];
end
labels = unique(labels);

clear IBPHMM_theta
K_est = length(labels);
IBPHMM_theta.K = K_est;
for k=1:K_est
    IBPHMM_theta.Mu(:,k)         = bestPsi.Psi.theta(labels(k)).mu;
    IBPHMM_theta.invSigma(:,:,k) = bestPsi.Psi.theta(labels(k)).invSigma \ eye(data.D);    
    IBPHMM_theta.Sigma(:,:,k)    = IBPHMM_theta.invSigma(:,:,k);
    IBPHMM_theta.invSigma_real(:,:,k) = bestPsi.Psi.theta(labels(k)).invSigma ; 
end

% Plot 2D HMM params
if size(Data{1},2) == 2
    if exist('h4','var') && isvalid(h4), delete(h4);end
    title_name  = 'Estimated Emission Parameters';
    plot_labels = {'$x_1$','$x_2$'};
    h4 = plotGaussianEmissions2D(IBPHMM_theta, plot_labels, title_name, labels);
end

%%%%%%%% Extract Sigmas %%%%%%%%%%%%
re_estimated = 0;
if re_estimated
    all_data   = [];
    all_labels = [];
    for d=1:length(Data)
        all_data   = [all_data Data{d}'];
        all_labels = [all_labels bestPsi.Psi.stateSeq(d).z];
    end
    [Priors0, Mu0, Sigma0] = gmmOracle(all_data, all_labels);
end
sigmas = [];
for k=1:IBPHMM_theta.K
    if re_estimated
        sigmas{k} = Sigma0(:,:,k);
    else
        sigmas{k} = IBPHMM_theta.invSigma(:,:,k);
    end
end
true_labels = [1:IBPHMM_theta.K];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%  Compute Similarity Matrix from B-SPCM Function for dataset %%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
dis_type = 2; 
gamma    = 5;
spcm = ComputeSPCMfunctionMatrix(sigmas, gamma, dis_type);  
D    = spcm(:,:,1);
S    = spcm(:,:,2);

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0); end
title_str = 'SPCM Similarity Matrix';
h0 = plotSimilarityConfMatrix(S, title_str);

if exist('h1','var') && isvalid(h1), delete(h1); end
title_str = 'SPCM Dis-similarity Matrix';
h1 = plotSimilarityConfMatrix(D, title_str);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%  Embed SDP Matrices in Approximate Euclidean Space  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
% Choose Embedding implementation
show_plots = 0;          % Show plot of similarity matrices+eigenvalues   
pow_eigen  = 4;          % (L^+)^(pow_eigen) for dimensionality selection 
[x_emb, Y, d_L_pow] = graphEuclidean_Embedding(S, show_plots, pow_eigen);
emb_name = '(SPCM) Graph-Subspace Projection';
M = size(Y,1);

%%%%%%%% Visualize Approximate Euclidean Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = emb_name; 
ml_plot_data(Y',plot_options);
axis equal;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Step 4: Discover Clusters of Covariance Matrices          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Discover Clusters with GMM-based Clustering Variants on Embedding %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0: sim-CRP-MM (Collapsed Gibbs Sampler) on Preferred Embedding
% 1: GMM-EM Model Selection via BIC on Preferred Embedding
% 2: CRP-GMM (Gibbs Sampler/Collapsed) on Preferred Embedding

est_options = [];
est_options.type             = 0;   % Clustering Estimation Algorithm Type   

% If algo 1 selected:
est_options.maxK             = 15;   % Maximum Gaussians for Type 1
est_options.fixed_K          = [];  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 300;   % Maximum Sampler Iterations
                                      % For type 0: 50-200 iter are needed
                                      % For type 2: 200-1000 iter are needed

% Plotting options
est_options.do_plots         = 1;              % Plot Estimation Stats
est_options.dataset_name     = dataset_name;   % Dataset name
est_options.true_labels      = true_labels;    % To plot against estimates

% Fit GMM to Trajectory Data
tic;
clear Priors Mu Sigma
[Priors, Mu, Sigma, est_labels, stats] = fitgmm_sdp(S, Y, est_options);
toc;

%%%%%%%%%% Compute Cluster Metrics %%%%%%%%%%%%%
[Purity, NMI, F, ARI] = cluster_metrics(true_labels, est_labels');
if exist('true_labels', 'var')
    K = length(unique(true_labels));
end
switch est_options.type
    case 0
        fprintf('---%s Results---\n Iter:%d, LP: %d, Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, ARI: %1.2f, F measure: %1.2f \n', ...
            'spcm-CRP-MM (Collapsed-Gibbs)', stats.Psi.Maxiter, stats.Psi.MaxLogProb, length(unique(est_labels)), K,  Purity, NMI, ARI, F);
    case 1
        fprintf('---%s Results---\n  Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, ARI: %1.2f, F measure: %1.2f \n', ...
            'Finite-GMM (MS-BIC)', length(unique(est_labels)), K,  Purity, NMI, ARI, F);
    case 2
        
        if isfield(stats,'collapsed')
            fprintf('---%s Results---\n  Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, ARI: %1.2f, F measure: %1.2f \n', ...
                'CRP-GMM (Collapsed-Gibbs)', length(unique(est_labels)), K,  Purity, NMI, ARI, F);
        else
            fprintf('---%s Results---\n  Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, ARI: %1.2f, F measure: %1.2f \n', ...
                'CRP-GMM (Gibbs)', length(unique(est_labels)), K,  Purity, NMI, ARI, F);
        end
end

%% Visualize Estimated Parameters
re_estimate = 1;
if M < 4     
    est_options.emb_name = emb_name;
    if re_estimate
        [Priors0, Mu0, Sigma0] = gmmOracle(Y, est_labels);
        tot_dilation_factor = 1; rel_dilation_fact = 0.2;
        Sigma0 = adjust_Covariances(Priors0, Sigma0, tot_dilation_factor, rel_dilation_fact);
        [~, est_labels0]       = my_gmm_cluster(Y, Priors0, Mu0, Sigma0, 'hard', []);
        [h_gmm, h_pdf]  = visualizeEstimatedGMM(Y,  Priors0, Mu0, Sigma0, est_labels0, est_options);
    else
        
        [h_gmm, h_pdf]  = visualizeEstimatedGMM(Y,  Priors, Mu, Sigma, est_labels, est_options);
    end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       Step 5: Extract Super States for Geometric Invariant Segmentation  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create super states from HMM emission model grouping
label_range = labels;
est_super_states = [];
for i=1:length(est_states)
    est_states_i = est_states{i};
    est_super_states_i = [];
    for j=1:length(est_states_i)
        est_super_states_i(j) = est_labels(find(est_states_i(j)==label_range));
    end
    est_super_states{i}    = est_super_states_i;
    est_super_states_3d{i} = est_super_states_i';
end
super_labels    = unique(est_labels);

% Plot 2D HMM params
if size(Data{1},2) == 2
    if exist('h4','var') && isvalid(h4), delete(h4);end
    title_name  = 'Estimated Emission Parameters';
    plot_labels = {'$x_1$','$x_2$'};
    h7 = plotGaussianEmissions2D(IBPHMM_theta, plot_labels, title_name, est_labels+4);
end

% Plot Segmentated 3D Trajectories
N = size(Data{1},2);
if N > 2        
    % Plot Estimated Segmentations
    if exist('h8','var') && isvalid(h8), delete(h8);end    
    title_name = {strcat(dataset_name,''),' $\mathcal{IBP}$-HMM + SPCM-$\mathcal{CRP}$-GMM'};
    title_edge = [1 0 0];
    h8 = plot3D_labeledTrajectories(Data,est_super_states_3d,plot_subsample,title_name, title_edge, dataset_view, dim);
end

% Plot Double Label Segmented Time-Series
if exist('h9','var') && isvalid(h2), delete(h9);end
title_name = strcat(dataset_name,' [$\mathcal{IBP}$-HMM + SPCM-$\mathcal{CRP}$-GMM ]');
h9 = plotDoubleLabelSegmentation2(Data, est_states, est_super_states, data_legends, title_name, label_range, est_labels);

% Compute Segmentation and State Clustering Metrics
ibpspcm_results = computeSegmClustmetrics(true_states_all, bestPsi, est_labels);
