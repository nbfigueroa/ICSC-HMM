%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Script for runnning evaulation of sequential model %%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Assume a set of IBP iterations have been run...
Chain_run = []; est_states = [];
% Segmentation Metrics
hamming_distance   = zeros(1,T);
global_consistency = zeros(1,T);
variation_info     = zeros(1,T);
K_S                = zeros(1,T);

% Clustering Metric Arrays
cluster_purity = zeros(1,T);
cluster_ARI    = zeros(1,T);
cluster_F      = zeros(1,T);
K_Z            = zeros(1,T);

for ii=1:T
    bestPsi = Best_Psi(ii);
    est_states = [];
    for e=1:length(bestPsi.Psi.stateSeq)
        est_states{e} = bestPsi.Psi.stateSeq(e).z';
    end
    Chain_run{ii}.est_states = est_states;
    
    % Extract Gaussian Emission Parameters
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
    
    Chain_run{ii}.IBPHMM_theta = labels;
    Chain_run{ii}.IBPHMM_theta = IBPHMM_theta;
        
    sigmas = [];
    for k=1:IBPHMM_theta.K
        if re_estimated
            sigmas{k} = Sigma0(:,:,k);
        else
            sigmas{k} = IBPHMM_theta.invSigma(:,:,k);
        end
    end
    true_labels = [1:IBPHMM_theta.K];
    Chain_run{ii}.sigmas = sigmas;
    Chain_run{ii}.true_labels = true_labels;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%  Compute Similarity Matrix from B-SPCM Function for dataset %%%%%%
    % %%%%%%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Tolerance for SPCM decay function
    dis_type = 2;
    gamma    = 5;
    spcm = ComputeSPCMfunctionMatrix(sigmas, gamma, dis_type);
    D    = spcm(:,:,1);
    S    = spcm(:,:,2);
    Chain_run{ii}.D = D;
    Chain_run{ii}.S = S;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%  Embed SDP Matrices in Approximate Euclidean Space  %%
    %%%%%%%%%%%%%%%%%%%%hamming_distance%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    show_plots = 0;          % Show plot of similarity matrices+eigenvalues
    pow_eigen  = 4;          % (L^+)^(pow_eigen) for dimensionality selection
    [x_emb, Y, d_L_pow] = graphEuclidean_Embedding(S, show_plots, pow_eigen);
    M = size(Y,1);
    Chain_run{ii}.Y = Y;
    
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
    est_options.samplerIter      = 100;   % Maximum Sampler Iterations
    est_options.do_plots         = 0;              % Plot Estimation Stats
    est_options.dataset_name     = dataset_name;   % Dataset name
    est_options.true_labels      = true_labels;    % To plot against estimates
    
    % Fit GMM to Trajectory Data
    tic;
    clear Priors Mu Sigma
    [Priors, Mu, Sigma, est_labels, stats] = fitgmm_sdp(S, Y, est_options);
    toc;
    clear SPCM_GMM
    SPCM_GMM.Priors = Priors;
    SPCM_GMM.Mu = Mu;
    SPCM_GMM.Sigma = Sigma;
        
    Chain_run{ii}.SPCM_GMM = SPCM_GMM;    
    Chain_run{ii}.est_labels = est_labels;   
    unique_labels = unique(est_labels);    
    for u=1:length(unique_labels)
        est_labels_mapped(find(est_labels==unique_labels(u))) = u;
    end
    
    % Compute Segmentation and State Clustering Metrics
    
    ibpspcm_results = computeSegmClustmetrics(true_states_all, bestPsi, est_labels_mapped);
    Chain_run{ii}.ibpspcm_results = ibpspcm_results;
    
    % Segmentation Metrics
    hamming_distance(ii)   = ibpspcm_results.hamming_distance_c;
    global_consistency(ii) = ibpspcm_results.global_consistency_c;
    variation_info(ii)     = ibpspcm_results.variation_info_c;
    K_S(ii)                = ibpspcm_results.inferred_states;

    % Clustering Metric Arrays
    cluster_purity(ii) = ibpspcm_results.cluster_purity;
    cluster_ARI(ii)    = ibpspcm_results.cluster_ARI;
    cluster_F(ii)      = ibpspcm_results.cluster_F;
    K_Z(ii)            = ibpspcm_results.inferred_state_clust;       
        
end
fprintf('*** IBP+SPCM Results*** \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) ARI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n   K_S: %3.3f (%3.3f) K_Z: %3.3f (%3.3f) \n',[mean(hamming_distance) std(hamming_distance)  ...
    mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_ARI) std(cluster_ARI) mean(cluster_F) std(cluster_F) mean(K_S) std(K_S)  mean(K_Z) std(K_Z) ])

