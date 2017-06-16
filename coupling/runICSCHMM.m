function [ChainHist] = runICSCHMM( dataParams, modelParams, outParams, algParams, initParams, saveDir )
% runBPHMM
% User-facing entry function for configuring and executing MCMC simulation 
%   for posterior inference of a Beta Process HMM (BP-HMM) model.
% INPUT:
%  Takes five arguments, each a cell array that specifies parameters
%   as Name/Value pairs, overriding default values (see defaults/ dir)
%  dataParams : 
%      params for data preprocessing (# of sequences, block-averaging, etc.)
%  modelParams :
%      params that define posterior of interest, like prior hyperparameters
%  outputParams :
%      specifies how often to save, how often to display, etc.
%      saveEvery, logPrEvery, printEvery, statsEvery, etc.
%  algParams :
%      MCMC alg behavior (# of iterations, parameters of proposal distr.)
%  initParams :
%   initial Markov state (# of initial features, initial state seq., etc.)
%   {'InitFunc', @initBPHMMfromGroundTruth'} initializes to known stateSeq

% Add required libraries, etc.
% configNPBayesToolbox;

% ================================================ SANITIZE INPUT
% Converts strings to doubles when possible, etc. to allow command line input
dataParams  = sanitizeUserInput( dataParams );
modelParams = sanitizeUserInput( modelParams );
outParams   = sanitizeUserInput( outParams );
algParams   = sanitizeUserInput( algParams );
initParams  = sanitizeUserInput( initParams );

% ================================================ INTERPRET INPUT
algDefs = defaultMCMCParams_BPHMM();
algParams = updateParamsWithUserInput(  algDefs, algParams );

outDefs = defaultOutputParams_BPHMM( outParams, algParams, saveDir );
outParams = updateParamsWithUserInput( outDefs, outParams(3:end) );

initDefs = defaultInitMCMC_BPHMM();
initParams = updateParamsWithUserInput( initDefs, initParams );

% ================================================= LOAD DATA
if isobject(dataParams)
    data = dataParams;
    Preproc = [];
else
    datasetName = dataParams{1};
    dataParams = dataParams(2:end);
    Preproc = getDataPreprocInfo( datasetName, dataParams );
    data = loadSeqData( datasetName, Preproc );
end

if outParams.doPrintHeaderInfo
    fprintf('Dataset Summary: \n\t %d time series items \n', data.N );
    fprintf('\t   # timesteps per seq:  min %d, median %.1f, max %d ) \n', min(data.Ts), median(data.Ts), max(data.Ts));
    fprintf('\t   %s\n', data.toString() );
end

model = defaultModelParams_BPHMM( data );
model = updateParamsWithUserInput( model, modelParams );

info_fname = fullfile( outParams.saveDir, 'Info.mat');
save( info_fname, 'data', 'Preproc', 'model', 'initParams', 'algParams', 'outParams' );

% ================================================= SET MCMC RAND NUM SEED
jobStr = num2str( outParams.jobID );
taskStr = num2str( outParams.taskID );
SEED = force2double( [jobStr taskStr] );

% MATLAB's seed (takes one integer)
SEED = mod( SEED, 2^32);
RandStream.setGlobalStream( RandStream( 'twister', 'Seed', SEED )   );
% Lightspeed toolbox seed (requires 3 integers)
randomseed( [SEED 1 2] );


fprintf('******** RUNNING COUPLED MODEL -- IBP Coupled SPCM-CRP HMM ********\n');


% ================================================= INITIALIZE MODEL
% Note: InitFunc will often use own random seed (reset internally only)
%   so that different sampling algs can be compared on *same* init state
[Psi, algParams, outParams] = initParams.InitFunc( data, model, initParams, algParams, outParams );
Psi.Z   = Psi.ThetaM.K;
Psi.K_z = length(unique(Psi.Z));

if outParams.doPrintHeaderInfo
    fprintf( '\t Emission Params : sampled from posterior given init stateSeq \n' );
    fprintf( 'Hyperparameters: \n' );
    fprintf( '\t            IBP mass param = %5.1f    ( resampling %d ) \n', model.bpM.gamma, algParams.BP.doSampleMass );
    fprintf( '\t            IBP conc param = %5.1f    ( resampling %d ) \n', model.bpM.c,     algParams.BP.doSampleConc );
    fprintf( '\t           HMM trans param = %5.1f    ( resampling %d ) \n', Psi.TransM.prior.alpha, algParams.HMM.doSampleHypers );
    fprintf( '\t    HMM trans sticky param = %5.1f    ( resampling %d ) \n', Psi.TransM.prior.kappa, algParams.HMM.doSampleHypers );
    fprintf( '\t        HMM emission param = %9s  \n', Psi.ThetaM.getParamDescr() );
    fprintf( '\t        Transform-Dependent Features (K_theta) = %d  \n',  Psi.ThetaM.K);
    fprintf( '\t        Transform-Invariant Features (K_phi)   = %d  \n',  Psi.K_z);
end



% ================================================= RUN INFERENCE
ChainHist = RunMCMCSimForICSCHMM( data, Psi, algParams, outParams);

end