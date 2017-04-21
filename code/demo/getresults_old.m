function [bestCH] = getresults_old(cellCH,s) 

logprobs = zeros(1,length(cellCH));
% for t=1:1:length(cellCH)
%     CH = cellCH{t};
%     logprobs(1,t) = CH.logPr(1,end).all;
% end

for t=1:1:length(cellCH)
    CH = cellCH{t};
    logs = zeros(1,length(CH.logPr));
    for l=1:1:length(CH.logPr)
        logs(1,l) = CH.logPr(1,l).all;
    end
    
    figure('Color',[1 1 1])
    plot(logs)
    xlabel('Markov Chain Iteration');
    ylabel('Joint Log Probability of Markov Chain State')
    
    [maxlog maxidx] = max(logs);   
    
    logprobs(1,t) = maxlog;
    logprobs(2,t) = maxidx;
end


logprobs

[Maxlp, Maxlp_trial] = max(logprobs(1,:));
Maxlp_iter = logprobs(2,Maxlp_trial);

bestCH.trial = Maxlp_trial;
bestCH.iter = Maxlp_iter;
bestCH.logPr = cellCH{Maxlp_trial}.logPr(1,Maxlp_iter);
ita = find(cellCH{Maxlp_trial}.iters.Psi<=cellCH{Maxlp_trial}.iters.logPr(1,Maxlp_iter));
itb = find(cellCH{Maxlp_trial}.iters.Psi>(cellCH{Maxlp_trial}.iters.logPr(1,Maxlp_iter)-s));

psi_iter = intersect(ita,itb);
bestCH.Psi = cellCH{Maxlp_trial}.Psi(1,psi_iter);


end