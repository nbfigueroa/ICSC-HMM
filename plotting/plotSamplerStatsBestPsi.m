function [h1, h1b, Best_Psi] = plotSamplerStatsBestPsi(Sampler_Stats, varargin)

% Gather High-level Stats
T = length(Sampler_Stats);
Iterations = Sampler_Stats(1).CH.iters.logPr;
estimated_feats  = zeros(1,T);
estimated_clusts = zeros(1,T);

Best_Psi = [];

% Plot Joint Log-prob and Estimated features/clusters
h1 = figure('Color',[1 1 1]);
if isfield(Sampler_Stats(1).CH.Psi(1), 'K_z')
    plots = 3;
else
    plots = 2;
end

subplot(plots,1,1)
for i=1:T
    joint_logs = zeros(1,length(Iterations));
    for ii=1:length(Iterations); joint_logs(1,ii) = Sampler_Stats(i).CH.logPr(ii).all;end
    [max_joint best_iter] = max(joint_logs);
    
    % Extract best iteration from each run    
    Best_Psi(i).logPr   = Sampler_Stats(i).CH.logPr(best_iter).all;
    [~, ids]  = sort(abs(Sampler_Stats(i).CH.iters.Psi - Sampler_Stats(i).CH.iters.logPr(best_iter)),'ascend');
    Best_Psi(i).Psi     = Sampler_Stats(i).CH.Psi(ids(1));
    Best_Psi(i).iter    = best_iter;
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
    color = [rand rand rand];
    semilogx(Iterations,joint_logs,'--*', 'LineWidth', 2,'Color',color); hold on;
    plot(best_iter, max_joint, 'o','MarkerFaceColor',color, 'MarkerSize', 10, 'MarkerEdgeColor',[0 0 0]);
end
xlim([1 Iterations(end)]);
xlabel('MCMC Iterations','Interpreter','LaTex','Fontsize',16); ylabel('LogPr','Interpreter','LaTex','Fontsize',20)

if isfield(Sampler_Stats(1).CH.Psi(1), 'K_z')
    title ({sprintf('Trace of Joint Probabilities $p(F, Z, S, X)$ for %d runs',[T])}, 'Interpreter','LaTex','Fontsize',20);
else    
    title ({sprintf('Trace of Joint Probabilities $p(F, S, X)$ for %d runs',[T])}, 'Interpreter','LaTex','Fontsize',20);
end
grid on

Iterations_feat = Sampler_Stats(1).CH.iters.Psi;

subplot(plots,1,2)
nFeats  = zeros(T,length(Iterations_feat));
nClusts = zeros(T,length(Iterations_feat));
for i=1:T    
    for ii=1:length(Iterations_feat); nFeats(i,ii) = length(Sampler_Stats(i).CH.Psi(ii).theta);end
    stairs(Iterations_feat, nFeats(i,:), 'LineWidth',2, 'Color', [rand/2 rand/2 1]); hold on;
    set(gca, 'XScale', 'log')
end
xlim([1 Iterations(end)])
xlabel('MCMC Iterations','Interpreter','LaTex','Fontsize',16); ylabel('$K$','Interpreter','LaTex','Fontsize',20)
title ({sprintf('Estimated features (shared states) for %d runs K: %3.1f (%3.1f) ',[T mean(estimated_feats) std(estimated_feats)])}, 'Interpreter','LaTex','Fontsize',20);
grid on

if isfield(Sampler_Stats(1).CH.Psi(1), 'K_z')
    subplot(plots,1,3)
    for i=1:T        
        for ii=1:length(Iterations_feat); nClusts(i,ii) = Sampler_Stats(i).CH.Psi(ii).K_z;end
        stairs(Iterations_feat, nClusts(i,:), 'LineWidth',2, 'Color', [1 rand/2 rand/2]); hold on;
        set(gca, 'XScale', 'log')
    end
    xlim([1 Iterations(end)])
    xlabel('MCMC Iterations','Interpreter','LaTex','Fontsize',16); ylabel('$K$','Interpreter','LaTex','Fontsize',20)
    title ({sprintf('Estimated state clusters for %d runs $K_z$: %3.1f (%3.1f) ',[T mean(estimated_clusts) std(estimated_clusts)])}, 'Interpreter','LaTex','Fontsize',20);
    grid on
end


%%% Plots of Hyper-parameter traces
h1b = figure('Color',[1 1 1]);
subplot(3,1,1)
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
title('Trace of IBP mass parameter $\gamma$','Interpreter','LaTex','Fontsize',20)


subplot(3,1,2)
for run=1:T       
    
    Iterations = Sampler_Stats(run).CH.iters.Psi;
    iters = length(Iterations);
    alphas = zeros(1,iters);
    for iter =1:iters
        alphas(1,iter) = Sampler_Stats(run).CH.Psi(iter).alpha;
    end
    
    % Plot joint traces
    semilogx(Iterations,alphas,'--*', 'LineWidth', 2,'Color',[rand rand rand]); hold on;   
    grid on
end
xlim([1 Iterations_feat(end)])
xlabel('MCMC Iterations', 'Interpreter','LaTex','Fontsize',16);
ylabel('$\alpha_b$', 'Interpreter','LaTex','Fontsize',20);
title('Trace of HMM concentration parameter $\alpha_b$','Interpreter','LaTex','Fontsize',20)


subplot(3,1,3)
for run=1:T       
    
    Iterations = Sampler_Stats(run).CH.iters.Psi;
    iters = length(Iterations);
    kappas = zeros(1,iters);
    for iter =1:iters
        kappas(1,iter) = Sampler_Stats(run).CH.Psi(iter).kappa;
    end
    
    % Plot joint traces
    semilogx(Iterations,kappas,'--*', 'LineWidth', 2,'Color',[rand rand rand]); hold on;   
    grid on
end
xlim([1 Iterations_feat(end)])
xlabel('MCMC Iterations', 'Interpreter','LaTex','Fontsize',16);
ylabel('$\kappa$', 'Interpreter','LaTex','Fontsize',20);
title('Trace of HMM sticky parameter $\kappa$','Interpreter','LaTex','Fontsize',20)


%%% Plot histogram of computed Features/Clusters
if ~isempty(varargin)
    h3 = figure('Color',[1 1 1]);
    if isfield(Sampler_Stats(1).CH.Psi(1), 'K_z')
        subplot(2,1,1)
        for f=1:size(nFeats,1);histogram(nFeats(f,:)); hold on; end
        grid on
        title({sprintf('Estimated Features $K$ throughout %d runs ',[T])},'Interpreter','LaTex','Fontsize',20)        
        
        subplot(2,1,2)
        for f=1:size(nClusts,1);histogram(nClusts(f,:)); hold on; end
        grid on
        title({sprintf('Estimated Features Clusters $K_Z$ throughout %d runs ',[T])},'Interpreter','LaTex','Fontsize',20)
    else
        for f=1:size(nFeats,1);histogram(nFeats(f,:)); hold on; end
        grid on
        title({sprintf('Estimated Features $K$ throughout %d runs ',[T])},'Interpreter','LaTex','Fontsize',20)
    end
end


end