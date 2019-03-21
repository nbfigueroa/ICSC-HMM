%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main demo scripts for the ICSC-HMM Segmentation Algorithm proposed in:
%
% N. Figueroa and A. Billard, “Transform-Invariant Non-Parametric Clustering 
% of Covariance Matrices and its Application to Unsupervised Joint Segmentation 
% and Action Discovery”
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
[~, Data, True_states] = genToyHMMData_Gaussian( N_TS, display ); 
label_range = unique(True_states{1});

%% 2a) Toy 2D dataset, 4 Unique Emission models, max 5 time-series
clc; clear all; close all;
M = 4; % Number of Time-Series
[~, ~, Data, True_states] = genToySeqData_Gaussian( 4, 2, M, 500, 0.5 ); 

%% 2b) Toy 2D dataset, 2 Unique Emission models transformed, max 4 time-series
clc; clear all; close all;
M = 3; % Number of Time-Series
[~, TruePsi, Data, True_states] = genToySeqData_TR_Gaussian(4, 2, M, 500, 0.5 );
dataset_name = '2D Transformed'; 

% Similarity matrix S (4 x 4 matrix)
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSimMat( TruePsi.S );

%% 5) Real 'Peeling' (max) 32-D dataset, 5 Unique Emission models, 3 time-series
% Demonstration of a Bimanual Peeling Task consisting of 
% 3 (32-d) time-series X = {x_1,..,x_T} with variable length T. 
%
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

% clc; 
clear all; close all
data_path = './test-data/'; display = 1; 

% Type of data processing
% O: no data manipulation -- 1: zero-mean -- 2: scaled by range * weights
normalize = 2; 

% Select dimensions to use
dim = 'robots'; 

% Define weights for dimensionality scaling
switch dim
    case 'active'
        weights = [5*ones(1,3) 2*ones(1,4) 1/10*ones(1,6)]'; % active    
    case 'act+obj'
        weights = [2*ones(1,7) 1/10*ones(1,6) 2*ones(1,6)]'; % act+obj
    case 'robots'
        weights = [5*ones(1,3) ones(1,4) 1/10*ones(1,6) ones(1,3) 2*ones(1,4) 1/20*ones(1,6) ]'; % robots(velocities)        
%         weights = [5*ones(1,3) 1/2*ones(1,4) 1/10*ones(1,6) ones(1,3) 1/2*ones(1,4) 1/20*ones(1,6) ]'; % robots(position)        
    case 'all'
        weights = [5*ones(1,3) ones(1,4) 1/10*ones(1,6) ones(1,3) 2*ones(1,4) 1/20*ones(1,6) 2*ones(1,6) ]'; % all        
end

% Define if using first derivative of pos/orient
use_vel = 0;

[data, TruePsi, Data, True_states, Data_] = load_peeling_dataset( data_path, dim, display, normalize, weights, use_vel);
dataset_name = 'Peeling';


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Step 1:Load One of the Wiping Datasets      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;
data_path  = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/ICSC-HMM/test-data/';

%%%%%%%%%%%  Input Data loading function %%%%%%%%%%
% Select Dataset to Load
dataset_type = 1; % 1: Rim Cover, 2: Door Fender                   
% Select which Variables to use
variables  = 1;
do_plots   = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data, TruePsi, Data, True_states, ...
    data_legends, dataset_name, plot_subsample, dataset_view] = load_wiping_dataset(data_path, dataset_type, do_plots, variables);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Step 1: Run E-M Model Selection for HMM with 10 runs in a range of K     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model Selection for HMM
K_range = [1:15]; repeats = 5; 
K_opt = hmm_eval(Data, K_range, repeats);

%  Fit HMM with 'optimal' K and Apply Viterbi for Segmentation
% Set "Optimal " GMM Hyper-parameters
K = K_opt; T = 5;
ts = [1:length(Data)];

% Segmentation Metric Arrays
hamming_distance   = zeros(1,T);
global_consistency = zeros(1,T);
variation_info     = zeros(1,T);

% Clustering Metric Arrays
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);

% Model Metric Arrays
logliks        = zeros(length(ts),T);
label_range    = [1:K];
est_states     = [];

for run=1:T
    % Fit HMM with 'optimal' K to set of time-series
    [p_start, A, phi, loglik] = ChmmGauss(Data, K);
        
    true_states_all   = [];
    est_states_all    = [];
    
    % Calculate p(X) & viterbi decode fo reach time-series
    if run==T
%         if exist('h0','var') && isvalid(h0), delete(h0);end
        h0 = figure('Color',[1 1 1]);
    end
    
    for i=1:length(ts)
        X = Data{ts(i)};
        
        if isfield(TruePsi, 'sTrueAll')
            true_states = TruePsi.s{ts(i)}';
        else
            true_states = True_states{ts(i)};
        end
        
        logp_xn_given_zn = Gauss_logp_xn_given_zn(Data{ts(i)}, phi);
        [~,~, logliks(i,run)] = LogForwardBackward(logp_xn_given_zn, p_start, A);
        est_states_ = LogViterbiDecode(logp_xn_given_zn, p_start, A);                       
        
        % Stack labels for state clustering metrics        
        true_states_all = [true_states_all; true_states];
        est_states_all  = [est_states_all; est_states_];
        
        % Plot segmentation Results on each time-series
        if run==T
            subplot(length(ts),1,i);
            data_labeled = [X est_states_]';
            plotLabeledData( data_labeled, [], strcat('HMM Segmented Time-Series (', num2str(ts(i)),'), K:',num2str(K),', loglik:',num2str(loglik)), [],label_range)
            est_states{i} = est_states_;
        end
    end
    
    % Segmentation Metrics per run
    [relabeled_est_states_all, hamming_distance(run),~,~] = mapSequence2Truth(true_states_all,est_states_all);
    [~,global_consistency(run), variation_info(run)] = compare_segmentations(true_states_all,est_states_all);
    
    % Cluster Metrics per run
    [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_states_all, est_states_all);
    
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               Extract Estimation Statistics         %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Overall Stats for HMM segmentation and state clustering
clc;
fprintf('*** Hidden Markov Model Results*** \n Optimal States: %d \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) NMI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n',[K mean(hamming_distance) std(hamming_distance)  ...
    mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])

% Visualize Transition Matrix and Segmentation from 'Best' Run
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotTransMatrix(A);

% Extract HMM Model parameters and Plot if 2D
% Extract params
HMM_theta = [];
HMM_theta.Mu = phi.mu;
HMM_theta.Sigma = phi.Sigma;
HMM_theta.K = K;

% Plot 2D HMM params
if size(Data{1},2) == 2
    title_name  = 'Estimated Emission Parameters';
    plot_labels = {'$x_1$','$x_2$'};
    if exist('h2','var') && isvalid(h2), delete(h2);end
    h2 = plotGaussianEmissions2D(HMM_theta, plot_labels, title_name);
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
    title_name = {dataset_name,'HMM (EM) Segmentation'};
    title_edge = [0.470588235294118 0.670588235294118 0.188235294117647];
    h6 = plot3D_labeledTrajectories(Data,est_states,plot_subsample,title_name, title_edge, dataset_view);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Step 5: Extract Super States for Geometric Invariant Segmentation      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% Extract Sigmas %%%%%%%%%%%%
sigmas = [];
for k=1:HMM_theta.K
    sigmas{k} = HMM_theta.Sigma(:,:,k);
end
true_labels = [1:HMM_theta.K];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%  Compute Similarity Matrix from B-SPCM Function for dataset %%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
dis_type = 2; 
gamma    = 3;
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
est_options.samplerIter      = 250;   % Maximum Sampler Iterations
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

%% Create super states from HMM emission model grouping
est_super_states    = [];
est_super_states_3d = [];
for i=1:length(est_states)
    est_states_i = est_states{i};
    est_super_states_i = [];
    for j=1:length(est_states_i)
        est_super_states_i(j) = est_labels(est_states_i(j));
    end
    est_super_states{i} = est_super_states_i;
    est_super_states_3d{i} = est_super_states_i';
end

super_labels    = unique(est_labels);

% Plot Segmentated 3D Trajectories
N = size(Data{1},2);

if N > 2        
    % Plot Estimated Segmentations
    if exist('h8','var') && isvalid(h8), delete(h8);end    
    title_name = {strcat(dataset_name,''),' HMM (EM) + SPCM-$\mathcal{CRP}$-GMM'};
    title_edge = [1 0 0];
    h8 = plot3D_labeledTrajectories(Data,est_super_states_3d,plot_subsample,title_name, title_edge, dataset_view);
end