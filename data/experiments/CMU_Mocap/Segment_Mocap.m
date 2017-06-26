%% [PRE_PROCESSING] Second Hands Ck-12 Environment Mocap Action Data 

%% Visualize Data
% Visualize the raw data time series my style
%   with background colored by "true" hidden state
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5], 'Color',[1 1 1] );
k = MocapData.N;
for i=k:k
%     subplot(k, 1, i );
    plotDataNadia( MocapData,i );
end

%% %
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5], 'Color',[1 1 1] );
for i=k:k
%     subplot(k, 1, i );
    plotData( MocapData,i );
end

%%
% Visualize the raw data time series
%   with background colored by "true" hidden state
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5],'Color',[1 1 1] );
subplot(6, 1, 1 );
plotData( MocapData, 1 );
subplot(6, 1, 2 );
plotData( MocapData, 2 );
subplot(6, 1, 3 );
plotData( MocapData, 3 );
subplot(6, 1, 4 ); 
plotData( MocapData, 4 );
subplot(6, 1, 5 );
plotData( MocapData, 5 );
subplot(6, 1, 6 );
plotData( MocapData, 6 );
%% PARAMETER SETTINGS FOR MCMC INFERENCE!

% Setting model hyper-parameters
modelP = {'bpM.gamma', 3,'obsM.Scoef',1,'bpM.c', 1};

% Sampling Algorithm Settings
algP = {'Niter', 300,'doSampleFUnique', 1, 'doSampleUniqueZ', 0, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven'};   %SM+DD

% Initial State of Markov Chain
initP  = {'F.nTotal', 1}; 

% Job IDs
jobIDAR  = 0305161;
jobIDGau = 0305162;

%% RUN MCMC INFERENCE
GauCH = {};
ARCH = {}; 
for ii=1:1:5
    % Start out with just one feature for all objects
    ARCH{ii} = runBPHMM( MocapDataAR, modelP, {jobIDAR, ii}, algP, initP );
    GauCH{ii} = runBPHMM( MocapData, modelP, {jobIDGau, ii}, algP, initP );
end

% ARCH/GauCH is a structure that captures the "Chain History" of the MCMC
%  it stores both model config at each each iteration (in Psi field)
%             and diagnostic information (log prob, sampler stats, etc.)

%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/AR-1 MODEL
    %Keep the trial with highest log likelihood 
% This one sucks most of the time %
[bestARCH] = getresults(ARCH,25);
fprintf('Model Type: AR(%d)\n',MocapDataAR.R);
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',bestARCH.trial, bestARCH.iter)
fprintf('Estimated Features: %d\n',length(bestARCH.Psi.theta));
fprintf('Log Likelihood: %f\n', bestARCH.logPr);
bestARPsi = bestARCH.Psi;

[Ar_Segm_results Ar_Total_feats] = plotSegDataNadia(MocapDataAR, bestARPsi, [1:MocapDataAR.N],'Best estimated AR-1 State Sequences' ,[])


%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/GAUSSIAN
%Keep the trial with highest log likelihood 
[bestGauCH] = getresults(GauCH,35);
fprintf('Model Type: Multivariate Gaussian\n');
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',bestGauCH.trial, bestGauCH.iter)
bestGauPsi = bestGauCH.Psi;
fprintf('Estimated Features: %d\n',length(bestGauPsi.theta));
fprintf('Log Likelihood: %f\n', bestGauCH.logPr);


[Gau_Segm_results Gau_Total_feats] = plotSegDataNadia(MocapData, bestGauPsi, [1:MocapData.N], 'Best estimated Gaussian State Sequence', []);


