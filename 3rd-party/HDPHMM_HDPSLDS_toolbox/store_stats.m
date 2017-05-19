% function S = store_stats(S,n,settings,stateSeq_n,dist_struct_n,theta_n,hyperparams_n)   
% Store statistics into structure S and save if rem(n,saveEvery) = 0

function S = store_stats(S,n,settings,stateSeq_n,dist_struct_n,theta_n,hyperparams_n)   

% If we are at a storing iteration:
if rem(n,settings.storeEvery)==0 & n>=settings.saveMin
    % And if we are at a mode-sequence storing iteration:
    if rem(n,settings.storeStateSeqEvery)==0
        % Store all sampled mode sequences:
        for ii=1:length(stateSeq_n)
            S.stateSeq(S.n,ii) = stateSeq_n(ii);
        end
        % Increment counter for the mode-sequence store variable:
        S.n = S.n + 1;
    end
    % Store all sampled model parameters:
    S.dist_struct(S.m) = dist_struct_n;
    S.theta(S.m) = theta_n;
    S.hyperparams(S.m) = hyperparams_n;
    % Increment counter for the regular store variable:
    S.m = S.m + 1;
    
end
    
% If we are at a saving iteration:
if rem(n,settings.saveEvery)==0

    % Save stats to specified directory:
    if isfield(settings,'filename')
        filename = strcat(settings.saveDir,'/',settings.filename,'iter',num2str(n),'trial',num2str(settings.trial));    % create filename for current iteration
    else
        filename = strcat(settings.saveDir,'/HDPHMMDPstats','iter',num2str(n),'trial',num2str(settings.trial));    % create filename for current iteration
    end

    save(filename,'S') % save current statistics
    
    % Reset S counter variables:
    S.m = 1;
    S.n = 1;
    
    display(strcat('Iteration: ',num2str(n)))
     
end