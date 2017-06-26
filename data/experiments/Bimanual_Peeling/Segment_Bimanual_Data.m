%% Apply Segmentation algorithm to bimanual independent data 

%% Visualize Data
% Visualize the raw data time series my style
%   with background colored by "true" hidden state
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5], 'Color',[1 1 1] );
for i=1:Active_Arm_Data.N
    subplot(Active_Arm_Data.N, 1, i );
    plotDataNadia( Active_Arm_Data,i );
    
end
suptitle('Active Arm')

figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5], 'Color',[1 1 1] );
for i=1:Passive_Arm_Data.N
    subplot(Passive_Arm_Data.N, 1, i );
    plotDataNadia( Passive_Arm_Data,i );    
end
suptitle('Passive Arm')
%% PARAMETER SETTINGS FOR MCMC INFERENCE!

% Setting model hyper-parameters
modelP = {'bpM.gamma', 4,'obsM.Scoef',1,'bpM.c',1}; 

% Sampling Algorithm Settings
algP = {'Niter', 300,'doSampleFUnique', 1, 'doSplitMerge', 1};  %SM works


% 
% Initial State of Markov Chain
initP  = {'F.nTotal', 5}; 

% Job IDs
jobIDAR  = 1005161;
jobIDGau = 1005162;

%% RUN MCMC INFERENCE
GauCH = {};
ARCH = {}; 
for ii=1:1:5
    % Start out with just one feature for all objects
    Active_GauCH{ii}   = runBPHMM( Active_Arm_Data, modelP, {jobIDGau, ii}, algP, initP );
    Passive_GauCH{ii} = runBPHMM( Passive_Arm_Data, modelP, {jobIDGau, ii}, algP, initP );
end

%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/GAUSSIAN on Active Arm Data
%Keep the trial with highest log likelihood 
[Act_bestGauCH] = getresults(Active_GauCH,35);
fprintf('Model Type: Multivariate Gaussian\n');
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',Act_bestGauCH.trial, Act_bestGauCH.iter)
Act_bestGauPsi = Act_bestGauCH.Psi;
fprintf('Estimated Features: %d\n',length(Act_bestGauPsi.theta));
fprintf('Log Likelihood: %f\n', Act_bestGauCH.logPr);

% Extract Recovered Sigmas
[Total_feats, sigmas] = extractSigmas(Active_Arm_Data, Act_bestGauPsi);

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
[Act_Gau_Segm_results Act_Gau_Total_feats] = plotSegDataNadia(Active_Arm_Data, Act_bestGauPsi, [1:Active_Arm_Data.N], 'Best estimated Gaussian State Sequence', groups);

% Extract Data Trajectories per Feature


%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/GAUSSIAN on Passive Arm Data
%Keep the trial with highest log likelihood 
[Pas_bestGauCH] = getresults(Passive_GauCH,35);
fprintf('Model Type: Multivariate Gaussian\n');
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',Pas_bestGauCH.trial, Pas_bestGauCH.iter)
Pas_bestGauPsi = Pas_bestGauCH.Psi;
fprintf('Estimated Features: %d\n',length(Pas_bestGauPsi.theta));
fprintf('Log Likelihood: %f\n', Pas_bestGauCH.logPr);

% Extract Recovered Sigmas
[Total_feats, sigmas] = extractSigmas(Passive_Arm_Data, Pas_bestGauPsi);

% Run SCPM-CRP to find sigma-clusters
options.tau = 1;
options.M   = 3;
cluster_labels = runSPCM_CRP(sigmas, options);
groups = {};          
for c=1:length(unique(cluster_labels)); 
    feat_id = cluster_labels == c;
    groups{1,c} = Total_feats(feat_id); 
end

[Pass_Gau_Segm_results Pass_Gau_Total_feats] = plotSegDataNadia(Passive_Arm_Data, Pas_bestGauPsi, [1:Passive_Arm_Data.N], 'Best estimated Gaussian State Sequence', groups);


