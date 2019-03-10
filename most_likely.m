% Most likely sequence
pi = bestPsi.pi{6}
est_labels_
z_0 = 4;

% Indicator Matrix, can make automaticallys from est_labels and pi
IZ = [1 0 0 0 0 0 0 ; 0 1 1 0 0 0 0 ; 0 0 0 1 0 1 1; 0 0 0 0 1 0 0]
IZ = [1 0 0 0 0 0  ; 0 1 1 0 0 0  ; 0 0 0 1 0 1 ; 0 0 0 0 1 0 ]
% IZ = [1 0 0 0 0 ; 0 1 1 0 0 ; 0 0 0 1 0 ; 0 0 0 0 1 ];

pi_new_    = (IZ*pi*IZ');

% Make at least right-stochastic
for i=1:length(pi_new_)
    pi_new_(:,i) = pi_new_(:,i)/sum(pi_new_(:,i));
end

% In log
% log_pi = log(pi_new_)
log_pi = pi_new_

% Compute possible paths
v = [1:length(unique(est_labels_))];
P = perms(v);
possible_paths = P(find(P(:,1)==z_0),:);

% Compute log probs of each path
logProbs = zeros(size(possible_paths,1),1);
for i=1:size(possible_paths,1)
    logProbs(i) = log_pi(possible_paths(i,1),possible_paths(i,1));
    for j=2:length(unique(est_labels_));
%         logProbs(i) = logProbs(i) + log_pi(possible_paths(i,j),possible_paths(i,j-1));
        logProbs(i) = logProbs(i) * log_pi(possible_paths(i,j),possible_paths(i,j-1));
    end
end

logProbs

[max_log max_id] = max(logProbs)
most_likely_path = possible_paths(max_id,:)
