function [Psi, Z_logPrb] = sampleFeatClusters(Psi)
old_Z = Psi.Z;
K_est = Psi.ThetaM.K;
if K_est >= 2        
    K_z = 1;
    while K_z == 1
        % Extract Sigmas
        sigmas = [];
        for k=1:K_est
            invSigma = Psi.ThetaM.theta(k).invSigma;
            sigma_ = invSigma \ eye(size(invSigma,1));
%             sigmas{k} = sigma_([1:3 8:13],[1:3 8:13]);
            sigmas{k} = sigma_;
        end
        
        % Settings and Hyper-Params for SPCM-CRP Clustering algorithm
        clust_options.tau           = 1;                % Tolerance Parameter for SPCM-CRP
        clust_options.type          = 'full';           % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
        clust_options.alpha         = randsample(4,1);  % Concentration parameter
        clust_options.plot_sim      = 0;
        clust_options.verbose       = 0;
        clust_options.T             = 15;      % Sampler Iterations
        if length(old_Z) ~= K_est
            clust_options.init_clust    = randsample(K_est,K_est)';
        else
            clust_options.init_clust    = old_Z;
        end
        
        % Mutliple Inference of SPCM-CRP Mixture Model
        runs = 3;
        parfor i=1:runs
            [Clust_Psi_{i}, ~, est_labels_{i}]  = run_SPCMCRP_mm(sigmas, clust_options);
        end
        
        MaxLogProb = -inf;
        for i=1:runs
            if Clust_Psi_{i}.MaxLogProb > MaxLogProb
                MaxLogProb = Clust_Psi_{i}.MaxLogProb;
                Clust_Psi  = Clust_Psi_{i};
                est_labels = est_labels_{i};
            end
        end        
        K_z = length(unique(est_labels));        
    end
    Psi.Z_logPrb = Clust_Psi.MaxLogProb;
else
    K_z = K_est;
    est_labels = 1:K_est;
end

Psi.Z        = est_labels;
Psi.K_z      = K_z;


end