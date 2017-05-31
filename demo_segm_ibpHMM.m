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
algP   = {'Niter', 100, 'HMM.doSampleHypers',0,'BP.doSampleMass',1,'BP.doSampleConc',1}; 

% Number of Repetitions
T = 10; 

% Run MCMC Sampler for T times
Sampler_Stats = [];
jobID = ceil(rand*1000);
for run=1:T       
    % Run Gibbs Sampler for Niter once.
    clear CH    
    % Start out with just one feature for all objects
    initP  = {'F.nTotal', randsample(data.N,1)}; 
    CH = runBPHMM( data, modelP, {jobID, run}, algP, initP, './IBP-Results' );  
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
for ii=1:T; log_probs(ii) = mean(Best_Psi(ii).logPr); end
[Max_prob, id] = max(log_probs);
bestPsi = Best_Psi(id)

% Extract info from 'Best Psi'
K_est = bestPsi.nFeats;
est_states_all = [];
for ii=1:data.N; est_states_all  = [est_states_all bestPsi.Psi.stateSeq(ii).z]; end
label_range = unique(est_states_all)

% Plot Segmentation
figure('Color',[1 1 1])
for i=1:data.N
    
    % Extract data from each time-series    
    X = data.Xdata(:,[data.aggTs(i)+1:data.aggTs(i+1)]);
    
    % Segmentation Direct from state sequence (Gives the same output as Viterbi estimate)
    est_states  = bestPsi.Psi.stateSeq(i).z;
    
    % Plot Inferred Segments
    subplot(data.N,1,i);
    data_labeled = [X; est_states];
    plotLabeledData( data_labeled, [], strcat('Segmented Time-Series (', num2str(i),'), K:',num2str(K_est),', logProb:',num2str(Max_prob)), {'x_1','x_2'},label_range)
    
end

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

%% Plot Estimated  Emission Parameters
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