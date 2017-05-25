function [anchorIDs, featIDs, PrQfeatIDs] = sampleAnchorAndFeatIDsToSplitMerge( Psi, data, algParams)
% Sample anchor ids and feature ids to perform split/merge move.
% Every split-merge move is defined by:
%   -- two *distinct* sequences (ii,jj), which we call "anchors"
%   -- two features, ki and kj, where F(ii,ki)=1 and F(jj,kj) = 1
% If ki == kj, we propose splitting these into two features,
%   otherwise, we merge ki and kj into a single feature.

F = Psi.F==1;
stateSeq = Psi.stateSeq;
ThetaM = Psi.ThetaM;

% ---------------------------------------------  select anchor sequences
if ~isfield( Psi, 'anchorIDs' ) 
    anchorIDs = randsample( data.N, 2 );
    else
    anchorIDs = Psi.anchorIDs;
end
ii = anchorIDs(1);
jj = anchorIDs(2);
% fprintf ('----- Anchor sequences: %d and %d -----\n', ii,jj);


if isfield( Psi, 'activeFeatIDs' )
    ki = Psi.activeFeatIDs(1);
    if length( Psi.activeFeatIDs ) > 1
        kj = Psi.activeFeatIDs(2);
    else
        kj = ki;
    end
end


% ---------------------------------------------  select feature IDs
switch algParams.SM.featSelectDistr
    case 'random'
        qs_ki = F(ii,:);
        if ~exist('ki','var')
            ki = multinomial_single_draw( qs_ki );
        end
        qs_kj = F(jj,:);
        if ~exist('kj','var')
            kj = multinomial_single_draw( qs_kj );
        end
        
        fprintf (' using random selection : %d and %d \n', ki,kj);
    case 'splitBias'
        qs_ki = F(ii,:);
        if ~exist('ki','var')
            ki = multinomial_single_draw( qs_ki );
        end
        
        delta_ki = false( size(F,2) );
        delta_ki( ki ) = 1;
        
        qs_kj = F(jj,:) .* ~delta_ki;
        qs_kj( ki ) = F(jj,ki)*2*sum( qs_kj );
        if ~exist('kj','var')
            kj = multinomial_single_draw( qs_kj );
        end
        
        fprintf (' using split bias : %d and %d \n', ki,kj);
    case 'splitBias+margLik'    
        % Build cond distr. kj | ki, jj
        %  based on margLik ratio between ki and kj
        
        qs_ki = F(ii,:);
        if ~exist('ki','var')
            ki = multinomial_single_draw( qs_ki );
        end
        
        log_qs_kj = -inf( 1, size(F,2)  );        
        
%         fprintf('Randomly chosen ki: %d',ki);
        
        for kk = find( F(jj,:) )
            if kk == ki
               continue;
            end                    
            log_qs_kj(kk) = ThetaM.calcMargLikRatio_MergeFeats( data, stateSeq, ki, kk );
        end             
        
        %Log-Marginal Likelihood Ratios      
        M = max( log_qs_kj );                
        if all( isinf(log_qs_kj) )
            % They are all the same
            qs_kj = zeros(1, size(F,2) );
            qs_kj(ki) = F(jj,ki);
        else            
            qs_kj = exp( log_qs_kj - M );
            qs_kj( ki ) = F(jj,ki)*2*sum( qs_kj ); 
            qs_kj = qs_kj ./ sum( qs_kj );        
        end
        
        % Final smoothing: take convex combo of 99% our qs and 1% us
        %   this avoids terrible reverse probabilities
        us = false( 1, size(F,2) );
        us( F(jj,:) ) = 1;
        us = us./sum(us);
        qs_kj = .99*qs_kj + 0.01*us;         
        
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% BLOCK OF CODE THAT USES SPCM TO FIND CANDIDATE %%%%%%%
        %%%%%%%%%%%%%%% FEATURE AND Q PROBABILITY %%%%%%%%%%%%%%%%%%%%
%         log_qs_kj_spcm = ones( 1, size(F,2)  )*(-inf);
%         tau = 10;
% 
%         Compute model-based pair-wise similarity
%         for kk = find( F(jj,:) )
%             if kk == ki
%                continue;
%             end                    
%             [p_sim, spcm, ~ , ~] = ComputeSPCMPair(ThetaM.theta(ki).invSigma^-1, ThetaM.theta(kk).invSigma^-1, tau);
%             log_qs_kj_spcm(kk) = log(p_sim); 
%         end  
%                 
%         M_spcm = max( log_qs_kj_spcm );        
%         if all( isinf(log_qs_kj_spcm) )
%             They are all the same
%             qs_kj_spcm = zeros(1, size(F,2) );
%             qs_kj_spcm(ki) = F(jj,ki);
%         else            
%             qs_kj_spcm = exp( log_qs_kj_spcm - M_spcm );
%             qs_kj_spcm( ki ) = F(jj,ki)*2*sum( qs_kj_spcm ); 
%             qs_kj_spcm = qs_kj_spcm ./ sum( qs_kj_spcm );        
%         end           
%         
%         
        if ~exist('kj','var')
            kj = multinomial_single_draw( qs_kj );            
%             kj_spcm = multinomial_single_draw( qs_kj_spcm );
%             kj = kj_spcm;
        end  
%         
%         Substitute qs_kj and kj computed by marg_like by spcm
%          qs_kj = qs_kj_spcm;                
end

% display('Proposal probabilities for each candidate:');
featIDs = [ki kj];
qs_ki = qs_ki ./ sum( qs_ki );
qs_kj = qs_kj ./ sum( qs_kj );

assert( ~any( isnan(qs_kj) ), 'ERROR: bad numerical calc of feat select distr.');

% Equation 5 of Hughes Paper
PrQfeatIDs = qs_ki( ki ) * qs_kj( kj );



