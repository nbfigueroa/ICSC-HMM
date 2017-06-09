function [Psi, Stats] = ICSCHMMsample( Psi, data, algP)
% Perform ONE iteration of sampling on the BPHMM model,
%  aggregating stats about sampler performance (accept rates) along the way
% Sweeps through the following MCMC moves:
%    shared features F     (MH updates, marg out z)
%    unique features F     (RJ birth/death, marg out z)
%    state sequences z     (Gibbs block sampling via dyn programming)
%    split merge     F,z   (sequential allocation proposals, MH accepted)
%    HMM trans.      eta   (Gibbs updates given z)
%    HMM emission    theta (Gibbs updates given z
%    HMM hypers      alph/kappa  (MH proposals via Gamma random walk)
%    BP  hypers      gamma/c     (MH proposals / Gibbs updates)
Stats=struct();
if algP.doAnneal ~= 0
   T0 = algP.Anneal.T0;
   Tf = algP.Anneal.Tf;
   
   if Psi.iter >= T0 && Psi.iter < Tf
       switch algP.doAnneal
        case 'Exp'
           tau = Tf/5; % 5*tau = "fully charged" (invTemp > 0.99 )
           Psi.invTemp = 1-exp(-(Psi.iter-T0)./tau);
        case 'Lin'
            Psi.invTemp = (Psi.iter-T0)/(Tf-T0);
       end
   elseif Psi.iter >= Tf
       Psi.invTemp = 1;
   else
       Psi.invTemp = 0;
   end
end

old_Z = Psi.Z;

if algP.doSampleFShared
    [Psi, Stats.FMH] = sampleSharedFeats( Psi, data );
end

if algP.doSampleFUnique
    [Psi, Stats.FRJ] = sampleUniqueFeats( Psi, data, algP, 0 );
end

if algP.doSampleZ
    Psi = sampleStateSeq( Psi, data );
    Psi.ThetaM = Psi.ThetaM.updateAllXSuffStats( horzcat(Psi.stateSeq(:).z), data );
end

if algP.doSplitMerge
    SM.ADD.nAccept=0;
    SM.ADD.nTotal =0;
    SM.DEL.nAccept=0;
    SM.DEL.nTotal =0;
    for trial = 1:algP.nSMTrials
        [nPsi, tS] = sampleSplitMerge_SeqAlloc( Psi, data, algP );
        Psi=nPsi;
        SM.(tS.moveDescr).nAccept = SM.(tS.moveDescr).nAccept + tS.nAccept;
        SM.(tS.moveDescr).nTotal  = SM.(tS.moveDescr).nTotal + 1;
    end
    Stats.SM = SM;
elseif algP.doSMNoQRev
    SM.ADD.nAccept=0;
    SM.ADD.nTotal =0;
    SM.DEL.nAccept=0;
    SM.DEL.nTotal =0;
    for trial = 1:algP.nSMTrials
        [nPsi, tS] = sampleSplitMerge_NoQRev( Psi, data, algP );
        Psi=nPsi;
        SM.(tS.moveDescr).nAccept = SM.(tS.moveDescr).nAccept + tS.nAccept;
        SM.(tS.moveDescr).nTotal  = SM.(tS.moveDescr).nTotal + 1;
    end
    Stats.SM = SM;
end

if algP.doSampleUniqueZ
    % Warning: after a successful accept,
    %   the thetas and etas held in "Psi" are no good!
    % MUST resample immediately.
    N = size(Psi.F,1);
    objIDs=randsample( 1:N, ceil(N/2) );
    [Psi, Stats.RJZ] = sampleUniqueFeats( Psi, data, algP, 1, objIDs );
end

if algP.doSampleEta
    Psi.TransM = Psi.TransM.sampleAllEta( Psi.F, Psi.stateSeq );
end

if algP.doSampleTheta
    Psi.ThetaM = Psi.ThetaM.sampleAllTheta( data, Psi.stateSeq );    
end

% Sampling feature clusters from current Theta estimate
K_est = Psi.ThetaM.K;
if K_est >= 2
    % Extract Sigmas
    sigmas = [];
    for k=1:K_est
        invSigma = Psi.ThetaM.theta(k).invSigma;
        sigmas{k} = invSigma \ eye(size(invSigma,1));
    end
    
    % Settings and Hyper-Params for SPCM-CRP Clustering algorithm
    clust_options.tau           = 1;      % Tolerance Parameter for SPCM-CRP
    clust_options.type          = 'full';  % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG    
    clust_options.alpha         = 1;       % Concentration parameter
    clust_options.plot_sim      = 0;
    clust_options.verbose       = 0;
    
    if length(old_Z) ~= K_est
        clust_options.T             = 10;      % Sampler Iterations
        clust_options.init_clust    = 1:K_est;
    else
        clust_options.T             = 2;      % Sampler Iterations
        clust_options.init_clust    = old_Z;
    end
    
    
    % Inference of SPCM-CRP Mixture Model
    [Clust_Psi, ~, est_labels]  = run_SPCMCRP_mm(sigmas, clust_options);
    K_z = length(unique(est_labels));
else
    K_z = K_est;
    est_labels = 1:K_est;
end

% Compute Ratios for Hyper-parameters
Psi.Z   = est_labels;
Psi.K_z = K_z;
M = length(Psi.stateSeq);
K_ratio = K_est/(K_z*M);
Psi.bpM.prior.a_mass = 0.5*K_ratio;
Psi.bpM.prior.b_mass = 0.5*K_ratio;

% Re-sample IBP Hyper-parameters
if algP.BP.doSampleMass || algP.BP.doSampleConc
    [Psi, Stats.BPconc] = sampleIBPhypers(Psi, algP);
end

% Re-sample HMM Hyper-parameters
if algP.HMM.doSampleHypers
    [Psi, Stats.HMMalpha, Stats.HMMkappa] = sampleHMMhypers( Psi, algP );
end



end % main function
