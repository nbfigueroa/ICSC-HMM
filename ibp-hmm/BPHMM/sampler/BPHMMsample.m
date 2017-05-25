function [Psi, Stats] = BPHMMsample( Psi, data, algP, dis, do_cluster)
% Perform ONE iteration of sampling on the BPHMM model,
%  aggregating stats about sampler performance (accept rates) along the way
% Sweeps through the following MCMC moves:
%    shared features F     (MH updates, marg out z)
%    unique features F     (RJ birth/death, marg out z)
%    state sequences z     (Gibbs block sampling via dyn programming)
%    split merge     F,z   (sequential allocation proposals, MH accepted)
%    HMM trans.      eta   (Gibbs updates given z)
%    HMM emission    theta (Gibbs updates given z)
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
    if dis
        display('Sampling Shared Features with MH')
    end
    
    [Psi, Stats.FMH] = sampleSharedFeats( Psi, data );
    if dis
        display('Result of Sampling Shared Features with MH')
    end    
end

if algP.doSampleFUnique
    if dis
        display('Sampling Unique Features with RJ')
    end
    [Psi, Stats.FRJ] = sampleUniqueFeats( Psi, data, algP, 0 );
    if dis
        display('Result of Sampling Unique Features with RJ')
    end
end

%-- Sampling dynamic parameters and transition variables the original way

if algP.doSampleZ
    if dis
        display('Sampling State Sequence with Gibbs block sampling');
    end    
    Psi = sampleStateSeq( Psi, data );
    Psi.ThetaM = Psi.ThetaM.updateAllXSuffStats( horzcat(Psi.stateSeq(:).z), data );   
    
end

%-- Contribition of Hughes et al. for faster BP-HMM inference with
% split-merge moves

if algP.doSplitMerge
    SM.ADD.nAccept=0;
    SM.ADD.nTotal =0;
    SM.DEL.nAccept=0;
    SM.DEL.nTotal =0;
    if dis
        display('Doing Sequentially Allocated Split-merge Updates w/MH Acceptance Probalities');
    end
    dis_ = 0;
    for trial = 1:algP.nSMTrials
        if dis_
            fprintf('--- Trial %d ---\n', trial);
        end
        [nPsi, tS] = sampleSplitMerge_SeqAlloc( Psi, data, algP, dis_);
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
        fprintf('Doing Split-merge with no Rev Q Updates on Sequence for the %d trial.\n',trial);
        [nPsi, tS] = sampleSplitMerge_NoQRev( Psi, data, algP );
        Psi=nPsi;
        SM.(tS.moveDescr).nAccept = SM.(tS.moveDescr).nAccept + tS.nAccept;
        SM.(tS.moveDescr).nTotal  = SM.(tS.moveDescr).nTotal + 1;
    end
    Stats.SM = SM;
end


if algP.doSampleUniqueZ
     fprintf('Resampling state sequence.\n');
    % Warning: after a successful accept,
    %   the thetas and etas held in "Psi" are no good!
    % MUST resample immediately.
    N = size(Psi.F,1);
    objIDs=randsample( 1:N, ceil(N/2) );
    [Psi, Stats.RJZ] = sampleUniqueFeats( Psi, data, algP, 1, objIDs );
    
end


if algP.doSampleEta
    if dis
        display('Sampling Transition Probabilities with Gibbs Updates')
    end
    Psi.TransM = Psi.TransM.sampleAllEta( Psi.F, Psi.stateSeq );
    
end

if algP.doSampleTheta
    if dis
        display('Sampling Dynamic Parameters (Theta) with Gibbs Updates')
    end
    Psi.ThetaM = Psi.ThetaM.sampleAllTheta( data, Psi.stateSeq );
    
end

% if algP.doClusterTheta
if do_cluster
    approx_thetas = {};
    if dis
        display('Sampling Clusters of Transformed Dynamic Parameteres (Theta) with MH')
    end     
    approx_thetas = Psi.ThetaM;    
    % Do the clustering here
    
    % Update Phi Models
%     Psi = sampleStateSeq( Psi, data );
%     Psi.PhiM = Psi.ThetaM.updateAllXSuffStats( horzcat(Psi.stateSeq(:).z), data );       
end


if algP.HMM.doSampleHypers
    if dis
        display('Sampling HMM HyperParams');
    end
    
    [Psi, Stats.HMMalpha, Stats.HMMkappa] = sampleHMMhypers( Psi, algP );
end

if algP.BP.doSampleMass || algP.BP.doSampleConc
    if dis
        display('Sampling BP HyperParams')
    end
    [Psi, Stats.BPconc] = sampleIBPhypers(Psi, algP);
end

% if algP.DP.doSampleMass 
%     if dis
%         display('Sampling DP HyperParams')
%     end
%     [Psi, Stats.BPconc] = sampleDPhypers(Psi, algP);
% end

end % main function
