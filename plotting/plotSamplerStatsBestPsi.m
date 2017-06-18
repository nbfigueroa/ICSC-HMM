function [h, Best_Psi] = plotSamplerStatsBestPsi(Sampler_Stats)

% Gather High-level Stats
T = length(Sampler_Stats);
Iterations = Sampler_Stats(1).CH.iters.logPr;
estimated_feats  = zeros(1,T);
estimated_clusts = zeros(1,T);

Best_Psi = [];

% Plot Joint Log-prob
h = figure('Color',[1 1 1]);
subplot(3,1,1)
for i=1:T
    joint_logs = zeros(1,length(Iterations));
    for ii=1:length(Iterations); joint_logs(1,ii) = Sampler_Stats(i).CH.logPr(ii).all;end
    [max_joint best_iter] = max(joint_logs);
    
    % Extract best iteration from each run
    Best_Psi(i).logPr = Sampler_Stats(i).CH.logPr(best_iter).all;
    [~, ids]  = sort(abs(Sampler_Stats(i).CH.iters.Psi - Sampler_Stats(i).CH.iters.logPr(best_iter)),'ascend');
    Best_Psi(i).Psi     = Sampler_Stats(i).CH.Psi(ids(1));
    Best_Psi(i).nFeats  = length(Best_Psi(i).Psi.theta);
    
    % Recollect Features 
    estimated_feats(1,i)  = Best_Psi(i).nFeats;
    
    if isfield(Sampler_Stats(1).CH.Psi,'K_z')
        Best_Psi(i).nClusts = Best_Psi(i).Psi.K_z;
        
        % Create cluster sequence and add it to best Psi struct
        stateSeq = Sampler_Stats(i).CH.Psi(ids(1)).stateSeq;
        for ss=1:length(stateSeq)
            clear c z
            z = stateSeq(ss).z;
            c = z;
            for k=1:length(Best_Psi(i).Psi.Z)
                c(z==k) = Best_Psi(i).Psi.Z(k);
            end
            stateSeq(ss).c = c;
        end
        Best_Psi(i).Psi.stateSeq = stateSeq;
        
        % Recollect Features and Clusters
        estimated_clusts(1,i) = Best_Psi(i).nClusts;
    end

    % Plot joint traces
    semilogx(Iterations,joint_logs,'--*', 'LineWidth', 2,'Color',[rand rand rand]); hold on;
end
xlim([1 Iterations(end)])
xlabel('MCMC Iterations','Interpreter','LaTex','Fontsize',16); ylabel('LogPr','Interpreter','LaTex','Fontsize',20)
title ({sprintf('Trace of Joint Probabilities $p(F, S, X)$ for %d runs',[T])}, 'Interpreter','LaTex','Fontsize',20)
grid on

Iterations_feat = Sampler_Stats(1).CH.iters.Psi;
subplot(3,1,2)
for i=1:T
    nFeats = zeros(1,length(Iterations_feat));
    for ii=1:length(Iterations_feat); nFeats(1,ii) = length(Sampler_Stats(i).CH.Psi(ii).theta);end
    
    stairs(Iterations_feat, nFeats, 'LineWidth',2, 'Color', [rand rand/2 1]); hold on;
    
    if isfield(Sampler_Stats(1).CH.Psi,'K_z')
        nClusts = zeros(1,length(Iterations_feat));
        for ii=1:length(Iterations_feat); nClusts(1,ii) = Sampler_Stats(i).CH.Psi(ii).K_z;end
        stairs(Iterations_feat, nClusts, 'LineWidth',2, 'Color', [1 rand rand/2]); hold on;
    end
    set(gca, 'XScale', 'log')
end
xlim([1 Iterations(end)])
xlabel('MCMC Iterations','Interpreter','LaTex','Fontsize',16); ylabel('$K$','Interpreter','LaTex','Fontsize',20)

if isfield(Sampler_Stats(1).CH.Psi,'K_z')
    title ({sprintf('Estimated features/clusters for %d runs $K$: %3.1f (%3.1f) and $K_z$: %3.1f (%3.1f) ',[T mean(estimated_feats) std(estimated_feats) mean(estimated_clusts) std(estimated_clusts)])}, 'Interpreter','LaTex','Fontsize',20);
else
    title ({sprintf('Estimated features (shared states) for %d runs K: %3.1f (%3.1f) ',[T mean(estimated_feats) std(estimated_feats)])}, 'Interpreter','LaTex','Fontsize',20);
end
grid on

subplot(3,1,3)
% figure('Color',[1 1 1])
for run=1:T       
    
    Iterations = Sampler_Stats(run).CH.iters.Psi;
    iters = length(Iterations);
    Gammas = zeros(1,iters);
    for iter =1:iters
        Gammas(1,iter) = Sampler_Stats(run).CH.Psi(iter).gamma;
    end
    
    % Plot joint traces
    semilogx(Iterations,Gammas,'--*', 'LineWidth', 2,'Color',[rand rand rand]); hold on;   
    grid on
end
xlim([1 Iterations_feat(end)])
xlabel('MCMC Iterations', 'Interpreter','LaTex','Fontsize',16);
ylabel('$\gamma$', 'Interpreter','LaTex','Fontsize',20);
title('$\gamma$ trace','Interpreter','LaTex','Fontsize',20)


end