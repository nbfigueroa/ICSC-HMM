function [labels energy] = kernel_kmeans(S, N_runs )
% % %%% Use Kernel-K-means to find the number of clusters from Similarity function  %%%%%%
    kk_means_labels = [];
    kk_means_energies = [];
    kk_means_K = [];
    D = size(S,1);    
    energy_threshold  = 0.5;
    
    for i = 1:N_runs
        k = min(poissonSample(D/2),D-2);
        if (k < 1)
            k = D-1;
        end
        [label, energy] = knkmeans(S,k);
        kk_means_labels(i,:) = label;
        kk_means_energies(i,:) = energy;
        kk_means_K(i,:) = length(unique(label));
    end
    
    exp_cluster_mean = round(mean(kk_means_K));
    
    % Using the "expected" cluster value
    K_exp = kk_means_K;
    idx_exp = find(K_exp==exp_cluster_mean);
    [val id] = min(kk_means_energies(idx_exp,:));    
    labels_mean = kk_means_labels(idx_exp(id),:);
    energy_mean = kk_means_energies(idx_exp(id),:);

    % Using the iteration with minimum Energy
    idx_zero = find(kk_means_energies==0);
    c_zero = kk_means_labels(idx_zero,:);
    clusters_zero = [];
    for c=1:size(c_zero,1)
        clusters_zero(c,:) = length(unique(c_zero(c,:)));
    end
    cluster_zero_not_total = find(clusters_zero < D);
    cluster_with_zero = 0;
    if ~isempty(idx_zero)
        if isempty(cluster_zero_not_total)            
            rand_id = randi(length(idx_zero),1);
            labels_min = kk_means_labels(idx_zero(rand_id),:);
            energy_min = kk_means_energies(idx_zero(rand_id),:);
        else
            rand_id = randi(length(cluster_zero_not_total),1);
            labels_ = kk_means_labels(cluster_zero_not_total(rand_id),:);
            energy_ = kk_means_energies(cluster_zero_not_total(rand_id),:);
            cluster_with_zero = 1;
        end
    end
    
    if ~(cluster_with_zero)
        c_non_zero = kk_means_energies;
        c_non_zero(idx_zero,:) = inf;
        min_energy = min(c_non_zero);
%         if min_energy < energy_threshold
            idx_min = find(c_non_zero==min_energy);
            rand_id = randi(length(idx_min),1);
%             labels_min = kk_means_labels(randsample(idx_min,1),:);
%             energy_min = kk_means_energies(randsample(idx_min,1),:);
            labels_min = kk_means_labels(idx_min(rand_id),:);
            energy_min = kk_means_energies(idx_min(rand_id),:);
%         else            
%             labels_min = labels_;
%             energy_min = energy_;        
%         end
    else
            labels_min = labels_;
            energy_min = energy_;        
    end
    
    if (energy_min < energy_mean) && (abs(energy_min-energy_mean) > 1e-3)
        energy = energy_min;
        labels = labels_min;
    else
        energy = energy_mean;
        labels = labels_mean;
    end
end