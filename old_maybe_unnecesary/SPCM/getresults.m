function [bestCH] = getresults(cellCH,s) 

logprobs = zeros(1,length(cellCH));
% for t=1:1:length(cellCH)
%     CH = cellCH{t};
%     logprobs(1,t) = CH.logPr(1,end).all;
% end

for t=1:1:length(cellCH)
    CH = cellCH{t};
    logs = zeros(1,length(CH.logPr));
    obs_logs = zeros(1,length(CH.logPr));
    for l=1:1:length(CH.logPr)
        logs(1,l) = CH.logPr(1,l).all;
        obs_logs(1,l) = CH.logPr(1,l).obs;
    end
    
    figure('Color',[1 1 1], 'Position',[1987 44 560 420])
    plot(CH.iters.logPr,logs,'Color',[1 0 0]);
    hold on
    plot(CH.iters.logPr,obs_logs,'Color',[0 0 1]);
    xlabel('Markov Chain Iteration');
    ylabel('Joint Log Probability of Markov Chain State')
    legend('Total Joint Prob.', 'Obs. Prob.')
    
    [maxlog maxidx] = max(logs);
    
    max_logprobs(1,t) = maxlog;
    max_logprobs(2,t) = maxidx;
end

max_logprobs;
[Maxlp, Maxlp_trial] = max(max_logprobs(1,:));
Maxlp_iter = max_logprobs(2,Maxlp_trial);

bestCH.logPr = Maxlp;
bestCH.trial = Maxlp_trial;
% bestCH.iter = Maxlp_iter;
bestCH.iter = cellCH{1,Maxlp_trial}.iters.logPr(Maxlp_iter);
% bestCH.logPr = cellCH{Maxlp_trial}.logPr(1,Maxlp_iter);
% ita = find(cellCH{Maxlp_trial}.iters.Psi<=cellCH{Maxlp_trial}.iters.logPr(1,Maxlp_iter));
% itb = find(cellCH{Maxlp_trial}.iters.Psi>(cellCH{Maxlp_trial}.iters.logPr(1,Maxlp_iter)-s));

% psi_iter = intersect(ita,itb);
% bestCH.Psi = cellCH{1,Maxlp_trial}.Psi(1,psi_iter);
bestCH.Psi = cellCH{1,Maxlp_trial}.Psi(1,Maxlp_iter);

end