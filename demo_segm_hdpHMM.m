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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Step 1 (a): Load One of the Wiping Datasets      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Step 1 (b): Load Rolling Datasets      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc; clear all; close all;
data_path  = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/ICSC-HMM/test-data/';
%%%%%%%%%%%  Input Data loading function %%%%%%%%%%
% Select Dataset to Load
variables   = 0;
do_plots    = 1;
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

clc;  clear all; 
%% close all
data_path  = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/ICSC-HMM/test-data/';

%%%%%%%%%%%  Input Data loading function %%%%%%%%%%
do_plots = 1; 
dim = 'robots'; 
time_series = 3; % Max is 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data, TruePsi, Data, True_states, ...
    data_legends, dataset_name, plot_subsample, dataset_view] = load_peeling_dataset2( data_path, do_plots, dim, time_series);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Step 2: Run Sticky HDP-HMM Sampler T times for good statistics   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define Settings for HDP-HMM %%%
clear hdp_options
hdp_options.obsModelType = 'Gaussian';
hdp_options.priorType = 'NIW';
hdp_options.d = size(Data{1},2);
hdp_options.sticky = 1;
hdp_options.kappa = 0.5;                    % NIW(kappa,theta,delta,nu_delta)
hdp_options.meanSigma = eye(hdp_options.d); % expected mean of IW(nu,nu_delta) prior on Sigma_{k,j}
hdp_options.Kz = 20;                        % truncation level of the DP prior on HMM transition distributions pi_k
hdp_options.Ks = 1;                         % truncation level of the DPMM on emission distributions pi_s (1-Gaussian emission)
hdp_options.plot_iter = 0;
hdp_options.Niter = 500;
hdp_options.saveDir = './hdp-Results';

%%% Create data structure of multiple time-series for HDP-HMM sampler %%%
clear data_struct 
for ii=1:length(Data)
    data_struct(ii).obs = Data{ii}';    
    % Set true_labels to visualize the sampler evolution
    data_struct(ii).true_labels = True_states{ii}';
end

%%%% Run Weak-Limit Gibbs Sampler for sticky HDP-HMM %%%
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

% Run Weak Limit Collapsed Gibbs Sampler for T times
ChainStats_Run = [];
for run=1:T       
    % Run Gibbs Sampler for Niter once.
    clear ChainStats
    ChainStats = run_HDPHMM_sampler(data_struct, hdp_options);
    
    % Extract Stats from Last Run
    ChainStats_Run(run).stateSeq  = ChainStats(end).stateSeq;
    ChainStats_Run(run).initProb  = ChainStats(end).dist_struct(end).pi_init;
    ChainStats_Run(run).TransProb = ChainStats(end).dist_struct(end).pi_z;
    ChainStats_Run(run).Theta     = ChainStats(end).theta(end);
    
    % Compute Log-Likelihood and est_states for this run
    true_states_all   = [];
    est_states_all    = [];
    loglik_ = zeros(length(data_struct),1);
    Theta.mu =  ChainStats_Run(run).Theta.mu;
    Theta.Sigma =  ChainStats_Run(run).Theta.invSigma;
    for jj=1:length(data_struct)
        logp_xn_given_zn = Gauss_logp_xn_given_zn(data_struct(jj).obs', Theta);
        [~,~, loglik_(jj,1)] = LogForwardBackward(logp_xn_given_zn, ChainStats_Run(run).initProb, ChainStats_Run(run).TransProb);
        
        % Stack labels for state clustering metrics   
        if isfield(TruePsi, 'sTrueAll')
            true_states = TruePsi.s{jj}';
        else
            true_states = True_states{jj};
        end
        
        est_states  = ChainStats_Run(run).stateSeq(jj).z';
        true_states_all = [true_states_all; true_states];
        est_states_all  = [est_states_all; est_states];
        
    end
    ChainStats_Run(run).logliks = loglik_;    
    
    % Segmentation Metrics per run
    [relabeled_est_states_all, hamming_distance(run),~,~] = mapSequence2Truth(true_states_all,est_states_all);
    [~,global_consistency(run), variation_info(run)] = compare_segmentations(true_states_all,est_states_all);
    inferred_states(run)   = length(unique(est_states_all));
    
    % Cluster Metrics per run
    [cluster_purity(run), cluster_NMI(run), cluster_F(run), cluster_ARI(run)] = cluster_metrics(true_states_all, relabeled_est_states_all);        
end

%% Overall Stats for HMM segmentation and state clustering
fprintf('*** Sticky HDP-HMM Results*** \n Optimal States: %3.3f (%3.3f) \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) ARI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n',[mean(inferred_states) std(inferred_states) mean(hamming_distance) std(hamming_distance)  ...
    mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_ARI) std(cluster_ARI) mean(cluster_F) std(cluster_F)])

% Make struct with metrics for comparison purposes
hdp_results = struct();
hdp_results.hamming_distance     = hamming_distance;
hdp_results.global_consistency   = global_consistency;
hdp_results.variation_info       = variation_info;
hdp_results.inferred_states      = inferred_states;
hdp_results.cluster_purity       = cluster_purity;
hdp_results.cluster_NMI          = cluster_NMI;
hdp_results.cluster_F            = cluster_F;
hdp_results.cluster_ARI          = cluster_ARI;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               Extract Estimation Statistics         %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize Transition Matrix, Emission Parameters and Segmentation from 'Best' Run
max_likelihoods  = zeros(1,T);
mean_likelihoods = zeros(1,T);
std_likelihoods  = zeros(1,T);
for ii=1:T 
    max_likelihoods(ii)  = max(ChainStats_Run(ii).logliks); 
    mean_likelihoods(ii) = mean(ChainStats_Run(ii).logliks); 
    std_likelihoods(ii)  = std(ChainStats_Run(ii).logliks); 
end
[Max_ll, max_id] = max(max_likelihoods)
[val_std id_std] = sort(std_likelihoods,'ascend');
[val_mean id_mean] = sort(mean_likelihoods,'descend');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              Extract Best Run and Plot Results            %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Segmentation with Chosen Run
id = id_mean(1)
id = max_id
BestChain = ChainStats_Run(id);
K_est = inferred_states(id);

% Plot Transition Matrix
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotTransMatrix(BestChain.TransProb);

% Plot Segmentation and Stats
if exist('h_hdp','var') && isvalid(h_hdp), delete(h_hdp);end
est_states_all = [];
for ii=1:length(data_struct); est_states_all  = [est_states_all BestChain.stateSeq(ii).z]; end
label_range = unique(est_states_all);
label_range_mapped = 1:length(label_range);
title_name = strcat(dataset_name,' [$\mathcal{HDP}$-HMM Segmentation]');
[handle, est_states_all, true_states_all, est_states, est_states_plot] = plotHDPLabelSegmentation(data_struct,BestChain,label_range, label_range_mapped, TruePsi, True_states, title_name, data_legends);


% Visualize Segmented Trajectories in 3D
N = size(Data{1},2);
if N > 2    
    % Plot Ground Segmentations
    if exist('h5','var') && isvalid(h5), delete(h5);end
    title_name = {dataset_name,'Ground Truth'};
    title_edge = [0 0 0];
    h5 = plot3D_labeledTrajectories(Data,True_states,plot_subsample,title_name, title_edge,  dataset_view, dim);    
    
    % Plot Estimated Segmentations
    if exist('h6','var') && isvalid(h6), delete(h6);end
    title_name = {dataset_name,' $\mathcal{HDP}$-HMM Segmentation'};
    title_edge = [0.470588235294118 0.670588235294118 0.188235294117647];
    h6 = plot3D_labeledTrajectories(Data,est_states_plot,plot_subsample,title_name, title_edge, dataset_view, dim);
end

% Segmentation Metrics per run
[relabeled_est_states_all, hamming_distance_,~,~] = mapSequence2Truth(true_states_all,est_states_all);
[~,global_consistency_, variation_info_] = compare_segmentations(true_states_all,est_states_all);
inferred_states_   = length(unique(est_states_all));

% Cluster Metrics per run
[cluster_purity_, cluster_NMI_, cluster_F_, cluster_ARI_] = cluster_metrics(true_states_all, relabeled_est_states_all);

% Overall Stats for HMM segmentation and state clustering
fprintf('\n*** Sticky HDP-HMM Results*** \n Optimal States: %3.3f \n Hamming-Distance: %3.3f GCE: %3.3f VO: %3.3f \n Purity: %3.3f  NMI: %3.3f  F: %3.3f   \n',[inferred_states_  hamming_distance_  ...
    global_consistency_ variation_info_ cluster_purity_ cluster_NMI_ cluster_F_])


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code below is to apply the spcm-crp-gmm clustering on the HDP-HMM
% emissions.. to show that the models learned with this algorithms are
% not adequate for such segmentation
% ==================================================>>
% ==================================================>>
% ==================================================>>
% ==================================================>>
% ==================================================>>

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Step 3: Create Vector Space Embeddings of HMM Emission Models    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract Emission Parameters
clear HDPHMM_theta
HDPHMM_theta.Mu = BestChain.Theta.mu(:,label_range);
HDPHMM_theta.invSigma = BestChain.Theta.invSigma(:,:,label_range);
HDPHMM_theta.K = K_est;
for k=1:HDPHMM_theta.K
    HDPHMM_theta.Sigma(:,:,k)    = HDPHMM_theta.invSigma(:,:,k) \ eye(size(Data{1},2));
end

% Plot 2D HMM params
if size(Data{1},2) == 2
    title_name  = 'Estimated Emission Parameters';
    plot_labels = {'$x_1$','$x_2$'};
    if exist('h2','var') && isvalid(h2), delete(h2);end
    h2 = plotGaussianEmissions2D(HDPHMM_theta, plot_labels, title_name);
end

%%%%%%%% Extract Sigmas %%%%%%%%%%%%
sigmas = [];
for k=1:HDPHMM_theta.K
    sigmas{k} = HDPHMM_theta.invSigma(:,:,k);
end
true_labels = [1:HDPHMM_theta.K];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%  Compute Similarity Matrix from B-SPCM Function for dataset %%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
dis_type = 2; 
gamma    = 4;
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
est_options.fixed_K          = 2;    % Fix K and estimate with EM for Type 1

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
%% Create super states from HMM emission model grouping
est_super_states = []; est_super_states_3d = []; 
for i=1:length(est_states)
    est_states_i = est_states{i};
    est_super_states_i = [];
    for j=1:length(est_states_i)
        est_super_states_i(j) = est_labels(find(est_states_i(j)==label_range_mapped));
    end
    est_super_states{i} = est_super_states_i;
    est_super_states_3d{i} = est_super_states_i';
end

super_labels    = unique(est_labels);

% Plot 2D HMM params
if size(Data{1},2) == 2
    if exist('h4','var') && isvalid(h4), delete(h4);end
    title_name  = 'Estimated Emission Parameters';
    plot_labels = {'$x_1$','$x_2$'};
    h7 = plotGaussianEmissions2D(HDPHMM_theta, plot_labels, title_name, est_labels+4);
end

% Plot Segmentated 3D Trajectories
N = size(Data{1},2);
if N > 2        
    % Plot Estimated Segmentations
    if exist('h8','var') && isvalid(h8), delete(h8);end    
    title_name = {strcat(dataset_name,''),' $\mathcal{HDP}$-HMM + SPCM-$\mathcal{CRP}$-GMM'};
    title_edge = [1 0 0];
    h8 = plot3D_labeledTrajectories(Data,est_super_states_3d,plot_subsample,title_name, title_edge, dataset_view);
end

% Plot Double Label Segmented Time-Series
if exist('h9','var') && isvalid(h9), delete(h9);end
title_name = strcat(dataset_name,' [$\mathcal{HDP}$-HMM + SPCM-$\mathcal{CRP}$-GMM]');
h9 = plotDoubleLabelSegmentation2(Data, est_states_plot, est_super_states_3d, data_legends, title_name, label_range_mapped, est_labels, 1);
