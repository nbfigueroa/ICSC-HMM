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


old_Z = Psi.Z;
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
Psi.Z = old_Z;


if algP.doSampleUniqueZ
    % Warning: after a successful accept,
    %   the thetas and etas held in "Psi" are no good!
    % MUST resample immediately.
    N = size(Psi.F,1);
    objIDs=randsample( 1:N, ceil(N/2) );
    [Psi, Stats.RJZ] = sampleUniqueFeats( Psi, data, algP, 1, objIDs );
end

% if algP.doSampleEta
    Psi.TransM = Psi.TransM.sampleAllEta( Psi.F, Psi.stateSeq );
% end

% if algP.doSampleTheta
    Psi.ThetaM = Psi.ThetaM.sampleAllTheta( data, Psi.stateSeq );    
% end

% Sampling feature clusters from current Theta estimate
[Psi] = sampleFeatClusters(Psi);

%%%% Compute Ratios for IBP Hyper-parameters %%%%
M = length(Psi.stateSeq);
K_ratio = Psi.ThetaM.K/(Psi.K_z*M);
coeff = 0.5;

Psi.bpM.prior.a_mass = coeff*K_ratio;
Psi.bpM.prior.b_mass = coeff*K_ratio;

% Re-sample IBP Hyper-parameterswrap o
if algP.BP.doSampleMass || algP.BP.doSampleConc
    [Psi, Stats.BPconc] = sampleIBPhypers(Psi, algP);
end

%%%% Compute Ratios for HMM Hyper-parameters %%%%
% Hyperparameters for prior on alpha:
Psi.TransM.prior.a_alpha = coeff*K_ratio;
Psi.TransM.prior.b_alpha = coeff*K_ratio;

% Variance of gamma proposal default --> var(alpha) = 2
algP.HMM.var_alpha = 2;

% Hyperparameters for prior on kappa:
K_kappa = Psi.ThetaM.K/Psi.K_z;
Psi.TransM.prior.a_kappa = 0.5*coeff*K_kappa;
Psi.TransM.prior.b_kappa = 0.5*coeff*K_kappa;

% Variance of gamma proposal default --> var(alpha) = 10
algP.HMM.var_kappa = 10;


% Re-sample HMM Hyper-parameters
if algP.HMM.doSampleHypers
    [Psi, Stats.HMMalpha, Stats.HMMkappa] = sampleHMMhypers( Psi, algP );
end



end % main function
