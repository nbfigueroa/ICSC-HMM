% Welcome!
% This easy demo first shows a simple toy BP-HMM dataset,
%  and then runs fast BP-HMM inference and visualizes the results!
% Make sure you've done these simple things to run this script:
%   -- install Eigen C++ library
%   -- install Lightspeed toolbox
%   -- Compile the MEX routines for fast sampling (./CompileMEX.sh)
%   -- Create local directories for saving results (./ConfigToolbox.sh)
% See QuickStartGuide.pdf in doc/ for details on configuring the toolbox

clear variables;
close all;

% First, we'll create some toy data
%   5 sequences, each of length T=500.
%   Each sequences selects from 4 behaviors, 
%     and switches among its selected set over time.
%     We'll use K=4 behaviors, each of which defines a distinct Gaussian
%     emission distribution (with 2 dimensions).
% Remember that data is a SeqData object that contains
%   the true state sequence labels for each time series

%[data, TruePsi] = genToySeqData_Gaussian(4, 8, 5, 500, 0.5); 
% [data] = genToySeqData_ARGaussian(4, 8, 5, 1000, 1); 
% [data] = genToySeqData_ARGaussian(4, 8, 5, -1, 1);
%%
% -------------------------------------------------   CREATE TOY DATA!
fprintf( 'Creating some robot toy data...\n' );
% [Robotdata] = genToyRobData(4,7,5);
%% Robot Toy Data
dataAR = RobotdataAR;
data = Robotdata;
%% Robot Toy Data w/Noise
dataAR = RobotdataARN;
data = RobotdataN;
%%
% Visualize the raw data time series my style
%   with background colored by "true" hidden state
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5], 'Color',[1 1 1] );
for i=1:data.N
    subplot(data.N, 1, i );
    plotDataNadia( data,i );
end
%%
% Visualize the raw data time series original style
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5] );
subplot(5, 1, 1 );
plotData( data, 1 );
subplot(5, 1, 2 );
plotData( data, 2 );
subplot(5, 1, 3 );
plotData( data, 3 );
subplot(5, 1, 4 );
plotData( data, 4 );
subplot(5, 1, 5 );
plotData( data, 5);


%%
% Visualize the "true" generating parameters
% Feat matrix F (binary 5 x 4 matrix )
figure('Units', 'normalized', 'Position', [0 0.5 0.3 0.5] );
plotFeatMat( TruePsi.F);
title( 'True Feature Matrix', 'FontSize', 20 );
%%
% Emission parameters theta (Gaussian 2D contours)
figure('Units', 'normalized', 'Position', [0.5 0.5 0.5 0.5] );
plotEmissionParams( TruePsi.theta, data );
title( 'True Emission Params (with all data points)', 'FontSize', 20 );

pause;
%%
% -------------------------------------------------  PARAMETER SETTINGS FOR MCMC INFERENCE!
%Test2-works kinda (No noise + Mean Only/No noise)
% modelP = {'bpM.gamma', 2, 'bpM.c',0.5}; %just gamma= 2 for workskinda/
% modelP = {'bpM.gamma', 2}; %just gamma= 2 for workskinda
% algP = {'doSampleFUnique', 1, 'doSplitMerge', 0, 'RJ.birthPropDistr', 'Prior'};   
% algP = {'Niter', 300,'doSampleFUnique', 0, 'doSplitMerge', 2};  %SM
% modelP = {'bpM.gamma', 5,'bpM.c', 1, 'hmmM.alpha' , 2,'hmmM.kappa',200};
% modelP = {'obsM.Scoef',5};
% algP   = {'Niter', 100, 'HMM.doSampleHypers',0,'BP.doSampleMass',0,'BP.doSampleConc',0}; 
% algP   = {'Niter', 100, 'HMM.doSampleHypers',0,'BP.doSampleMass',0,'BP.doSampleConc',0,'theta.birthPropDistr', 'DataDriven'}; %DD
% algP   = {'Niter', 100, 'HMM.doSampleHypers',0,'BP.doSampleMass',0,'BP.doSampleConc',0,'doSplitMerge', 1, 'doSampleFUnique', 0};%SM
% algP   = {'Niter', 100, 'HMM.doSampleHypers', 1,'BP.doSampleMass', 1,'BP.doSampleConc', 1,'doSplitMerge', 1, 'doSampleFUnique', 0};%SM

% TESTS (Jan 12, 2014)
modelP = {'bpM.gamma', 3,'obsM.Scoef',1,'bpM.c', 1}; %for rotation invariance (loose)
% NEW TESTS (Nov 19, 2013)
modelP = {'bpM.gamma', 2,'obsM.Scoef',1,'bpM.c', 1}; %original settings

%For hesitation data
modelP = {'bpM.gamma', 10,'obsM.Scoef',2,'bpM.c', 1}; %original settings
% TESTING LOOSEN STICKINESS
% modelP = {'bpM.gamma', 10,'obsM.Scoef',1,'bpM.c', 0.5,'hmmM.alpha', 1, 'hmmM.kappa',2};

% algP = {'doSampleFUnique', 1, 'doSplitMerge', 0, 'RJ.birthPropDistr', 'Prior'};  
algP = {'Niter', 100,'doSampleFUnique', 0, 'doSplitMerge', 1};  %SM works
% algP = {'Niter', 100,'doSampleFUnique', 1, 'doSplitMerge', 1,'RJ.birthPropDistr', 'DataDriven'};  %SM+DD works
% algP = {'Niter', 100,'doSampleFUnique', 1, 'doSplitMerge', 1,'doSampleHypers',1}; 
% algP = {'doSampleFUnique', 1, 'doSplitMerge', 0, 'RJ.birthPropDistr', 'Prior'}; %Prior  

% algP = {'Niter', 300, 'doSampleFUnique', 1, 'doSampleUniqueZ', 0, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven'};  %SM+DD
% algP = {'Niter', 300,'doSampleFUnique', 0, 'doSampleUniqueZ', 1, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven'}; %SM+zDD 
% algP = {'Niter', 300, 'doSplitMerge',1};
% algP = {'Niter', 300,'doSplitMerge',1 };

%Like Mocap experiment
modelP = {};
algP = {'Niter', 100,'doSampleFUnique', 1, 'doSampleUniqueZ', 0, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven'};   %SM+DD
% jobID  = [10 11 12];
initP  = {'F.nTotal', 1}; 
% jobID = randsample(10,1);
jobIDAR  = 1205151;
jobIDGau = 1205152;

%Testing Algo
%algP = {'Niter', 10,'doSampleFUnique', 0, 'doSplitMerge', 1};  %SM works
%initP  = {'F.nTotal', 1}; 

%% RUN MCMC INFERENCE
GauCH = {};
ARCH = {}; 
for ii=1:1:5
% algP   = {'Niter', 300, 'HMM.doSampleHypers',0,'BP.doSampleMass',0,'BP.doSampleConc',0}; 
% Start out with just one feature for all objects
ARCH{ii} = runBPHMM( dataAR, modelP, {jobIDAR, ii}, algP, initP );
GauCH{ii} = runBPHMM( data, modelP, {jobIDGau, ii}, algP, initP );
end
% CH = runBPHMM( data, modelP, {jobID, taskID, 'printEvery', 25, 'doPrintHeaderInfo', 0}, algP, initP );

% CH is a structure that captures the "Chain History" of the MCMC
%  it stores both model config at each each iteration (in Psi field)
%             and diagnostic information (log prob, sampler stats, etc.)

%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/AR-1 MODEL
    %Keep the trial with largest log likelihood 
% [bestARCH, worstARCH] = getresults(ARCH);
[bestARCH] = getresults(ARCH,5);
fprintf('Model Type: AR(%d)\n',dataAR.R);
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',bestARCH.trial, bestARCH.iter)
fprintf('Estimated Features: %d\n',length(bestARCH.Psi.theta));
fprintf('Log Likelihood: %f\n', bestARCH.logPr.all);
bestARPsi = bestARCH.Psi;
[Segm_results Total_feats] = plotSegDataNadia(data, bestARPsi, [1:data.N])
% plotStateSeq(bestARPsi,dataAR)
% plotSegDataNadia(data, bestARPsi, [1 4 8 12 16])
% title('Best estimated AR State Sequence')
%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/GAUSSIAN
[bestGauCH] = getresults(GauCH,5);
fprintf('Model Type: Multivariate Gaussian\n');
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',bestGauCH.trial, bestGauCH.iter)
bestGauPsi = bestGauCH.Psi;
fprintf('Estimated Features: %d\n',length(bestGauPsi.theta));
fprintf('Log Likelihood: %f\n', bestGauCH.logPr.all);

% Toy Dataset
[Segm_results Total_feats] = plotSegDataNadia(data, bestGauPsi, [1:data.N]);
%[Segm_results Total_feats] = plotSegDataNadia(data, bestGauPsi, [1:12]);
% Carrot Grating
%  [Segm_results Total_feats] = plotSegDataNadia(data, bestGauPsi, [1:15])

%% EXTRACT REAL SEGMENTED TRAJECTORIES
ExtSeg = {};
for i=1:length(Total_feats)
%     fprintf('---Behav %d ---\n',Total_feats(i));
    l = 1;
    for j=1:length(Segm_results)
        segments = Segm_results{j};
        seg_id = find(segments(:,1)==Total_feats(i));
        if ~isempty(seg_id)
%           fprintf('-->Demo %d \n',j);
          for k=1:length(seg_id)
              if seg_id(k)==1
                  start = 1;
              else
                  start = segments(seg_id(k)-1,2);
              end
              finish = segments(seg_id(k),2);
%               fprintf('From %d to %d \n', start, finish)
              seg.demo = j;
              seg.start = start;
              seg.finish = finish;     
              ExtSeg{i,l} = seg;
              l = l+1;
          end 
        end
    end
end

%% Results for test trial
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Estimated Features: %d\n',length(CH.Psi(end).theta));
fprintf('Log Likelihood: %f\n', CH.logPr(end).all);
bestPsi = CH.Psi(end);
plotStateSeq(bestPsi)
title('Best estimated AR State Sequence')


fprintf('Model Type: Multivariate Gaussian\n');
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Estimated Features: %d\n',length(GauCH.Psi(end).theta));
fprintf('Log Likelihood: %f\n', GauCH.logPr(end).all);
bestPsi = GauCH.Psi(end);
% plotStateSeq(bestPsi,data)

%%
% -------------------------------------------------   VISUALIZE RESULTS!
% Remember: the actual labels of each behavior are irrelevent
%   so there won't in general be direct match with "ground truth"
% For example, the true behavior #1 may be inferred behavior #4

% So we'll need to align recovered parameters (separately at each iter)
% Let's just look at iter 90 and iter 100

GauPsi100 = GauCH.Psi( GauCH.iters.Psi == 100 );
GaualignedPsi10 = alignPsiToTruth_OneToOne( GauPsi100, data );
plotStateSeq(GauPsi100,data)
%%
Psi100 = CH.Psi( CH.iters.Psi == 100 );
alignedPsi100 = alignPsiToTruth_OneToOne( Psi100, data );
plotStateSeq(Psi100,data)
%%
% Estimated feature matrix F
% figure( 'Units', 'normalized', 'Position', [0 0.5 0.5 0.5] );
% subplot(1,2,1);
figure;plotFeatMat( alignedPsi90 )
title( 'F (@ iter 90)', 'FontSize', 20 );
% subplot(1,2,2);
figure;plotFeatMat( alignedPsi100 );
title( 'F (@ iter 100)', 'FontSize', 20 );
%%
% Estimated emission parameters
% figure( 'Units', 'normalized', 'Position', [0.5 0.5 0.5 0.5] );
% subplot(1,2,1);
% plotEmissionParams( Psi90 );
% title( 'Theta (@ iter 90)', 'FontSize', 20 );
% subplot(1,2,2);

figure;plotEmissionParams( Psi100, CH);
title( 'Theta (@ iter 100)', 'FontSize', 20 );
%%
% Estimated state sequence
plotStateSeq( alignedPsi10, [1 2 3 4 5] );
set( gcf, 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5] );
title('Est. Z : Seq 3', 'FontSize', 20 );

%%
plotStateSeq( alignedPsi100, [1 2 3 4 5] );
set( gcf, 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5] );
title('Est. Z : Seq 3', 'FontSize', 20 );

fprintf( 'Remember: actual labels for behaviors are *irrelevant* from model perspective\n');
fprintf( '  what matters: *aligned* behaviors consistently assigned to same datapoints as ground truth\n' );