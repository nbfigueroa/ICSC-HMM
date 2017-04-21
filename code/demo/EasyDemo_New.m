% Welcome!
% This easy demo first shows a simple toy BP-HMM dataset,
%  and then runs fast BP-HMM inference and visualizes the results!
% Make sure you've done these simple things to run this script:
%   -- install Eigen C++ library
%   -- install Lightspeed toolbox
%   -- Compile the MEX routines for fast sampling (./CompileMEX.sh)
%   -- Create local directories for saving results (./ConfigToolbox.sh)
% See QuickStartGuide.pdf in doc/ for details on configuring the toolbox
clear all;
clear variables;
close all;

% -------------------------------------------------   CREATE TOY DATA!
fprintf( 'Creating some toy data...\n' );
% First, we'll create some toy data
%   5 sequences, each of length T=500.
%   Each sequences selects from 4 behaviors, 
%     and switches among its selected set over time.
%     We'll use K=4 behaviors, each of which defines a distinct Gaussian
%     emission distribution (with 2 dimensions).
% Remember that data is a SeqData object that contains
%   the true state sequence labels for each time series
[data, TruePsi] = genToySeqData_Gaussian( 4, 2, 5, 500, 0.5 ); 
% [data, TruePsi] = genToySeqData_Gaussian_null( 5, 2, 5, 500, 0.5 ); 

%%
% Visualize the raw data time series my style
%   with background colored by "true" hidden state
% figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5], 'Color',[1 1 1] );
% for i=1:data.N
%     subplot(data.N, 1, i );
%     plotDataNadia( data,i );
% end

%%
% Visualize the raw data time series
%   with background colored by "true" hidden state
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5],'Color',[1 1 1] );
subplot(5, 1, 1 );
plotData( data, 1 );
subplot(5, 1, 2 );
plotData( data, 2 );
subplot(5, 1, 3 );
plotData( data, 3 );
subplot(5, 1, 4 );
plotData( data, 4 );
subplot(5, 1, 5 );
plotData( data, 5 );

%%
% Emission parameters theta (Gaussian 2D contours)
figure('Units', 'normalized',  'Position', [0.5 0.5 0.5 0.5],'Color',[1 1 1] );
plotEmissionParams( TruePsi.theta, data );
title( 'True Emission Params (with all data points)', 'FontSize', 20 );
axis

sigmas = {};
for i = 1:length(TruePsi.theta)
    sigmas{i} = TruePsi.theta(i).invSigma^-1;
end

use_log = 1;
% Compute Similarity
spcm = ComputeSPCMfunction(sigmas, use_log);  
figure('Color', [1 1 1]);        
imagesc(spcm(:,:,1))
title('log(SPCM) Confusion Matrix')
colormap(copper)
colormap(pink)
colorbar 

% Find Anchor pair
spcm_pairs = nchoosek(1:1:size(spcm,1),2);
for i=1:size(spcm_pairs,1)
        spcm_pairs(i,3) = spcm(spcm_pairs(i,1),spcm_pairs(i,2),1);
end

% pause;
%%
% -------------------------------------------------   RUN MCMC INFERENCE!
modelP = {'bpM.gamma', 2}; 
algP   = {'Niter', 100, 'HMM.doSampleHypers',0,'BP.doSampleMass',0,'BP.doSampleConc',0}; 

%MoCap Data
modelP = {'bpM.gamma', 2,'obsM.Scoef',1,'bpM.c', 1};
algP = {'Niter', 1,'doSampleFUnique', 1, 'doSampleUniqueZ', 0, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven'};   %SM+DD

jobIDGau = ceil(rand*1000);
GauCH = {};
ARCH = {}; 
for ii=1:1
% Start out with just one feature for all objects
initP  = {'F.nTotal', 1}; 
GauCH{ii} = runBPHMM( data, modelP, {jobIDGau, ii}, algP, initP );
end


%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/GAUSSIAN
[bestGauCH] = getresults(GauCH,5);
fprintf('Model Type: Multivariate Gaussian\n');
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',bestGauCH.trial, bestGauCH.iter)
bestGauPsi = bestGauCH.Psi(end);
fprintf('Estimated Features: %d\n',length(bestGauPsi.theta));
fprintf('Log Likelihood: %f\n', bestGauCH.logPr.all);

% Toy Dataset
[Segm_results Total_feats] = plotSegDataNadia(data, bestGauPsi, 1:data.N);

% Estimated emission parameters
figure( 'Units', 'normalized', 'Position', [0.5 0.5 0.5 0.5], 'Color', [1 1 1] );
plotEmissionParams(bestGauPsi);
title( 'Recovered Emission Parameters', 'FontSize', 20 );

% Check Similarity of Recovered Features
rec_thetas = bestGauPsi.theta(Total_feats);
rec_sigmas = {};
for i = 1:length(rec_thetas)
    rec_sigmas{i} = rec_thetas(i).invSigma^-1;
end

% Compute Similarity
use_log = 0;
spcm = ComputeSPCMfunction(rec_sigmas, use_log);  
figure('Color', [1 1 1]);        
imagesc(spcm(:,:,1))
title('log(SPCM) Confusion Matrix')
colormap(copper)
colormap(pink)
colorbar 

