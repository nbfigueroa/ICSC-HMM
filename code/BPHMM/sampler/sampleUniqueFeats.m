function [Psi, Stats, RhoTerms] = sampleUniqueFeats( Psi, data, algParams, doZMove, objIDs )
% Sample *unique* features for each time series, 
%   using reversible jump moves that propose adding/deleting unique feat.
%   from each sequence.
% The proposal distribution for reversible jump is defined in "algParams'
%   can be any of:
%      'prior'  : emit param thetaStar drawn from prior
%      'data-driven' : emit param thetaStar draw from data posterior
%OUTPUT
%  Psi : resulting model config
%  Stats : struct that summarizes reversible jump performance
%           counts # birth (ADD) and death (DEL) attempts and acceptances

if ~exist( 'objIDs', 'var')
    objIDs = 1:data.N;
end


Stats.ADD.nAccept = 0;
Stats.ADD.nTotal = 0;
Stats.DEL.nAccept = 0;
Stats.DEL.nTotal = 0;

for ii = objIDs
    if doZMove
        fprintf('Doing Z Move as well');
        [Psi, RhoTerms] = sampleSingleFeat_UniqueRJStateSeq( ii, Psi, data, algParams );
    else
%         fprintf('Using %s proposal distribution \n',algParams.RJ.birthPropDistr)        
        [Psi, RhoTerms] = sampleSingleFeatEntry_UniqueRJ( ii, Psi, data, algParams );
    end
    
    if RhoTerms.doBirth
        if RhoTerms.doAccept
%             fprintf('Time-series %d ... added a feature with %f log acceptance. prob \n',ii, RhoTerms.log_rho_star);
        end
        Stats.ADD.nTotal = Stats.ADD.nTotal+1;
        if RhoTerms.doAccept;
            Stats.ADD.nAccept = Stats.ADD.nAccept+1;
        end
        
    else
        if RhoTerms.doAccept
%             fprintf('Time-series %d ...deleted features with %f log acceptance. prob \n',ii, RhoTerms.log_rho_star);
        end
        Stats.DEL.nTotal = Stats.DEL.nTotal+1;
        if RhoTerms.doAccept;
            Stats.DEL.nAccept = Stats.DEL.nAccept+1;
        end
    end
end
