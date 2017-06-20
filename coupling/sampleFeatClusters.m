function [Psi] = sampleFeatClusters(Psi)
old_Z = Psi.Z;
K_est = Psi.ThetaM.K;
if K_est >= 2    
    K_z = 1;
    while K_z == 1
        % Extract Sigmas
        sigmas = [];
        for k=1:K_est
            invSigma = Psi.ThetaM.theta(k).invSigma;
            sigmas{k} = invSigma \ eye(size(invSigma,1));
        end
        
        % Settings and Hyper-Params for SPCM-CRP Clustering algorithm
        clust_options.tau           = 1;       % Tolerance Parameter for SPCM-CRP
        clust_options.type          = 'full';  % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
        clust_options.alpha         = 1;       % Concentration parameter
        clust_options.plot_sim      = 0;
        clust_options.verbose       = 0;
                
        if length(old_Z) ~= K_est            
            clust_options.T             = 10;      % Sampler Iterations
            clust_options.init_clust    = randsample(K_est,K_est)';
        else
            clust_options.T             = 2;      % Sampler Iterations
            clust_options.init_clust    = old_Z;
        end
        
        % Inference of SPCM-CRP Mixture Model
        [Clust_Psi, ~, est_labels]  = run_SPCMCRP_mm(sigmas, clust_options);
        K_z = length(unique(est_labels));
    end
else
    K_z = K_est;
    est_labels = 1:K_est;
end

Psi.Z        = est_labels;
Psi.K_z      = K_z;
Psi.Z_logPrb = Clust_Psi.MaxLogProb;

end