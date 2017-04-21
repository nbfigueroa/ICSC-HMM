%% Apply Segmentation algorithm to joint bimanual data 

%% Visualize Data
% Visualize the raw data time series my style
%   with background colored by "true" hidden state
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5], 'Color',[1 1 1] );
for i=1:Bimanual_Arm_Data.N
    subplot(Bimanual_Arm_Data.N, 1, i );
    plotDataNadia( Bimanual_Arm_Data,i );
    
end
suptitle('Dual-arm + Object Data')

%% PARAMETER SETTINGS FOR MCMC INFERENCE!

% Setting model hyper-parameters
modelP = {'bpM.gamma', 4,'obsM.Scoef',1,'bpM.c',1}; 

% Sampling Algorithm Settings
algP = {'Niter', 300,'doSampleFUnique', 1, 'doSplitMerge', 1};  %SM works

% Initial State of Markov Chain
initP  = {'F.nTotal', 5}; 

% Job IDs
jobIDAR  = 1105161;
jobIDGau = 1105162;

%% RUN MCMC INFERENCE
GauCH = {};
ARCH = {}; 
for ii=1:1:5
    % Start out with just one feature for all objects
    Bimanual_GauCH{ii}   = runBPHMM( Bimanual_Arm_Data, modelP, {jobIDGau, ii}, algP, initP );
end

%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/GAUSSIAN on Active Arm Data
%Keep the trial with highest log likelihood 
[Bi_bestGauCH] = getresults(Bimanual_GauCH,35);
fprintf('Model Type: Multivariate Gaussian\n');
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',Bi_bestGauCH.trial, Bi_bestGauCH.iter)
Bi_bestGauPsi = Bi_bestGauCH.Psi;
fprintf('Estimated Features: %d\n',length(Bi_bestGauPsi.theta));
fprintf('Log Likelihood: %f\n', Bi_bestGauCH.logPr);

% Extract Recovered Sigmas
[Total_feats, sigmas] = extractSigmas(Bimanual_Arm_Data, Bi_bestGauPsi);

% Run SCPM-CRP to find sigma-clusters
options.tau = 1;
options.M   = 3;
cluster_labels = runSPCM_CRP(sigmas, options);
groups = {};          
for c=1:length(unique(cluster_labels)); 
    feat_id = cluster_labels == c;
    groups{1,c} = Total_feats(feat_id); 
end

% Visualize Segmentation and Sigma-Clustering
[Bi_Gau_Segm_results Bi_Gau_Total_feats] = plotSegDataNadia(Bimanual_Arm_Data, Bi_bestGauPsi, [1:Bimanual_Arm_Data.N], 'Best estimated Gaussian State Sequence', groups);

