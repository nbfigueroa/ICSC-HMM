function [h, Best_Psi] = plotSamplerStatsBestPsi(Sampler_Stats)

% Gather High-level Stats
T = length(Sampler_Stats);
Iterations = Sampler_Stats(1).CH.iters.logPr;
estimated_feats = zeros(1,T);

Best_Psi = [];

% Plot Joint Log-prob
h = figure('Color',[1 1 1])
subplot(2,1,1)
for i=1:T
    joint_logs = zeros(1,length(Iterations));
    for ii=1:length(Iterations); joint_logs(1,ii) = Sampler_Stats(i).CH.logPr(ii).all;end
    [max_joint best_iter] = max(joint_logs);
    
    % Extract best iteration from each run
    Best_Psi(i).logPr = Sampler_Stats(i).CH.logPr(best_iter).all;
    [~, ids]  = sort(abs(Sampler_Stats(i).CH.iters.Psi - Sampler_Stats(i).CH.iters.logPr(best_iter)),'ascend');
    Best_Psi(i).Psi    = Sampler_Stats(i).CH.Psi(ids(1));
    Best_Psi(i).nFeats = length(Best_Psi(i).Psi.theta);
    estimated_feats(1,i) = Best_Psi(i).nFeats;
    
    % Plot joint traces
    semilogx(Iterations,joint_logs,'--*', 'LineWidth', 2,'Color',[rand rand rand]); hold on;
end
xlim([1 Iterations(end)])
xlabel('MCMC Iteration','Interpreter','LaTex','Fontsize',20); ylabel('LogPr','Interpreter','LaTex','Fontsize',20)
title ({sprintf('Trace of Joint Probabilities $p(F, S, X)$ for %d runs',[T])}, 'Interpreter','LaTex','Fontsize',20)
grid on

Iterations_feat = Sampler_Stats(1).CH.iters.Psi;
subplot(2,1,2)
for i=1:T
    nFeats = zeros(1,length(Iterations_feat));
    for ii=1:length(Iterations_feat); nFeats(1,ii) = length(Sampler_Stats(i).CH.Psi(ii).theta);end
    
    stairs(Iterations_feat,nFeats, 'LineWidth',2); hold on;
    set(gca, 'XScale', 'log')
    xlim([1 Iterations_feat(end)])
end
xlim([1 Iterations_feat(end)])
xlabel('MCMC Iteration','Interpreter','LaTex','Fontsize',20); ylabel('$K$','Interpreter','LaTex','Fontsize',20)
title ({sprintf('Estimated features (shared states) for %d runs K: %3.1f (%3.1f) ',[T mean(estimated_feats) std(estimated_feats)])}, 'Interpreter','LaTex','Fontsize',20)
grid on


end