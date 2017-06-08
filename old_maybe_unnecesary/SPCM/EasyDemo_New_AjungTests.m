% Welcome!
% This easy demo first shows a simple toy BP-HMM dataset,
%  and then runs fast BP-HMM inference and visualizes the results!
% Make sure you've done these simple things to run this script:
%   -- install Eigen C++ library
%   -- install Lightspeed toolbox
%   -- Compile the MEX routines for fast sampling (./CompileMEX.sh)
%   -- Create local directories for saving results (./ConfigToolbox.sh)
% See QuickStartGuide.pdf in doc/ for details on configuring the toolbox
clear all;
clear variables;
% close all;

% -------------------------------------------------   CREATE TOY DATA!
fprintf( 'Creating some toy data...\n' );
% First, we'll create some toy data
%   5 sequences, each of length T=500.
%   Each sequences selects from 4 behaviors, 
%     and switches among its selected set over time.
%     We'll use K=4 behaviors, each of which defines a distinct Gaussian
%     emission distribution (with 2 dimensions).
% Remember that data is a SeqData object that contains
%   the true state sequence labels for each time series
[data, TruePsi] = genToySeqData_Gaussian( 4, 3, 5, 500, 0.5 ); 
% [data, TruePsi] = genToySeqData_Gaussian_null( 5, 2, 5, 500, 0.5 ); 

%%
% Visualize the raw data time series my style
%   with background colored by "true" hidden state
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5], 'Color',[1 1 1] );
for i=1:data.N
    subplot(data.N, 1, i );
    plotDataNadia( data,i );
end

%%
% Visualize the raw data time series
%   with background colored by "true" hidden state
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5],'Color',[1 1 1] );
subplot(5, 1, 1 );
plotData( data, 1 );
subplot(5, 1, 2 );
plotData( data, 2 );
subplot(5, 1, 3 );
plotData( data, 3 );
subplot(5, 1, 4 ); 
plotData( data, 4 );
subplot(5, 1, 5 );
plotData( data, 5 );

%% TRUE DATA REPRESENTATION
clc
% close all

if ~isempty(data.zTrueAll) || exist('TruePsi')
    % %%%%%% Generate GROUND TRUTH state sequences %%%%%%%%%%%%%%
    
    if exist('TruePsi')
        if ~isfield(TruePsi,'stateSeq')
            TruePsi.stateSeq = {};
            for i=1:data.N
                z_ids = [data.aggTs(1,i) + 1 data.aggTs(1,i+1)]; 
                z = data.zTrueAll(z_ids(1):z_ids(2));
                TruePsi.stateSeq(1,i).z = z;
            end
        end
    end
    

    if isfield(TruePsi,'theta')
        % %%%%%% Generate Sigmas for SPCM-bases Evaluation %%%%%%%%%%%%%%
        sigmas = {};
        for i = 1:length(TruePsi.theta)
        %     sigmas{i} = cov2cor(TruePsi.theta(i).invSigma^-1);
            sigmas{i} = TruePsi.theta(i).invSigma^-1;
        end

        % %%%%%% Visualize Prob. Similarity Confusion Matrix %%%%%%%%%%%%%%
        tau = 5;
        spcm = ComputeSPCMfunctionProb(sigmas, tau);  
        figure('Color', [1 1 1], 'Position',[ 3283  545  377 549]);           

        subplot(3,1,1)
        imagesc(spcm(:,:,2))
        title('Probability of Similarity Confusion Matrix')
        colormap(pink)
        colorbar 

        % % %%% Use Kernel-K-means to find the number of clusters from Similarity function  %%%%%%
        N_runs = 100;
        [labels_kkmeans energy] = kernel_kmeans(log(spcm(:,:,2)), N_runs);
        K = length(unique(labels_kkmeans));

        fprintf('After SPCM and KK-means--->>> \n Number of clusters: %d with total energy %d\n', K, energy);
        subplot(3,1,2)
        imagesc(labels_kkmeans)
        title('Clustering from kernel-kmeans on SPCM Prob. function')
        axis equal tight
        colormap(pink)

        % %%% Compute clusters from Similarity Matrix using Affinity Propagation %%%%%%
        subplot(3,1,3)
        spcm_aff = log(spcm(:,:,2));
        prob_spcm_aff = diag(median(spcm_aff,2)) + spcm_aff;
        [E K labels_aff idx] = affinitypropagation(prob_spcm_aff);
        fprintf('After Affine Propagation on SPCM Prob. function--->>> \n Number of clusters: %d\n', K);
        imagesc(labels_aff')
        title('Clustering from Aff. Prop. on SPCM Prob. function')
        axis equal tight
        colormap(pink)

        if energy > 1e-2
            labels = labels_aff;
        else
            labels = labels_kkmeans;
        end

        % %%%%%% Generate GROUND TRUTH feature grouping %%%%%%%%%%%%%%
        groups = {};          
        for c=1:K; groups{1,c} = find(labels==c); end
        
        % %%%%%% Visualize GROUND TRUTH segmentation, features and groupings %%%%%%%%%%%%%%
        [Segm_results Total_feats] = plotSegDataNadia(data, TruePsi, 1:data.N, 'Real state sequence', groups);
        
        
        % %%%%%% Create cluster label sequence %%%%%%%
        for ii=1:data.N
            Zest = TruePsi.stateSeq(1,ii).z;
            Cest = Zest;            
            cluster_labels = 1:length(groups);
            cluster_labels = cluster_labels + Total_feats(end);

            for k=1:length(groups)
                cluster_label = cluster_labels(k);
                group = groups{k};
                for kk=1:length(group)            
                    Cest(find(Zest == group(kk))) = cluster_label;
                end               
            end
            TruePsi.stateSeq(1,ii).c = Cest;
        end        
    else
        display('Displaying segmentation ground truth z only');
        % %%%%%% Visualize GROUND TRUTH segmentation, features and groupings %%%%%%%%%%%%%%
        [Segm_results Total_feats] = plotSegDataNadia(data, TruePsi, 1:data.N, 'Real state sequence', []);
    
    end
end
%%
% -------------------------------------------------   RUN MCMC INFERENCE!
clc
close all
% modelP = {'bpM.gamma', 2}; 
% algP   = {'Niter', 100, 'HMM.doSampleHypers',0,'BP.doSampleMass',0,'BP.doSampleConc',0}; 
% algP = {'Niter', 1,'doSampleFUnique', 1, 'doSplitMerge', 1};
% TESTS (Jan 12, 2014)
modelP = {'bpM.gamma', 3,'obsM.Scoef',1,'bpM.c', 1}; %for rotation invariance (loose)

% For hesitation data testing
% modelP = {'bpM.gamma', 3,'obsM.Scoef',1,'bpM.c', 1,'hmmM.alpha' , 2,'hmmM.kappa',2000};


% algP = {'Niter', 100,'doSampleFUnique', 0, 'doSplitMerge', 1};  %SM works
algP = {'Niter', 100,'doSampleFUnique', 1, 'doSampleUniqueZ', 0, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven'};   %SM+DD

jobIDGau = ceil(rand*1000);
jobIDAR = ceil(rand*1000);
GauCH = {};

ii = 1;
% for ii=1:1
% Start out with just one feature for all objects
% Start out with just one feature for all objects
initP  = {'F.nTotal', 1}; 
GauCH{ii} = runBPHMM( data, modelP, {jobIDGau, ii}, algP, initP);
% ARCH{ii} = runBPHMM( dataAR, modelP, {jobIDAR, ii}, algP, initP );
% end

%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/GAUSSIAN
% clc
close all
clear bestGauCH
clear bestGauPsi

[bestGauCH] = getresults(GauCH,1);
fprintf('Model Type: Multivariate Gaussian\n');
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',bestGauCH.trial, bestGauCH.iter)
bestGauPsi = bestGauCH.Psi;
fprintf('Estimated Features: %d\n',length(bestGauPsi.theta));
fprintf('Log Likelihood: %f\n', bestGauCH.logPr);

% Check Similarity of Recovered Features
max_feats = 0;
Zests = [];
iis = 1:data.N;
for i=1:length(iis)
    ii = iis(i);
    Zest = bestGauPsi.stateSeq(1,ii).z;
    Zests = [Zests unique(Zest)];
    if  max(Zest)>max_feats
        max_feats = max(Zest);
    end
end

Total_feats = unique(Zests);

rec_thetas = bestGauPsi.theta(Total_feats);
rec_sigmas = {};
for i = 1:length(rec_thetas)
    rec_sigmas{i} = rec_thetas(i).invSigma^-1;
end

% %%%%%% Visualize Prob. Similarity Confusion Matrix %%%%%%%%%%%%%%
tau = 5;
spcm = ComputeSPCMfunctionProb(rec_sigmas, tau);  
figure('Color', [1 1 1], 'Position',[ 3283  545  377 549]);           

subplot(3,1,1)
imagesc(spcm(:,:,2))
title('Probability of Similarity Confusion Matrix')
colormap(pink)
colorbar 

% % %%% Use Kernel-K-means to find the number of clusters from Similarity function  %%%%%%
% N_runs = 100;
% [labels_kkmeans energy] = kernel_kmeans(log(spcm(:,:,2)), N_runs);
% K = length(unique(labels_kkmeans));

% fprintf('After SPCM and KK-means--->>> \n Number of clusters: %d with total energy %d\n', K, energy);
% subplot(3,1,2)
% imagesc(labels_kkmeans)
% title('Clustering from kernel-kmeans on SPCM Prob. function')
% axis equal tight
% colormap(pink)

% %%% Compute clusters from Similarity Matrix using Affinity Propagation %%%%%%
subplot(3,1,3)
spcm_aff = log(spcm(:,:,2));
prob_spcm_aff = diag(median(spcm_aff,2)) + spcm_aff;
[E K labels_aff idx] = affinitypropagation(prob_spcm_aff);
fprintf('After Affine Propagation on SPCM Prob. function--->>> \n Number of clusters: %d\n', K);
imagesc(labels_aff')
title('Clustering from Aff. Prop. on SPCM Prob. function')
axis equal tight
colormap(pink)

% if length(unique(labels_aff))==1                    
%     labels = labels_kkmeans;                                            
% else
    labels = labels_aff;
% end
    
% %%%%%% Generate Estimated feature grouping %%%%%%%%%%%%%%
groups = {};          
for c=1:K; groups{1,c} = find(labels==c); end

% Toy Dataset
[Segm_results Total_feats] = plotSegDataNadia(data, bestGauPsi, 1:data.N, 'Best recovered state sequence',groups);


% %%%%%% Create cluster label sequence %%%%%%%
for ii=1:data.N
    Zest = bestGauPsi.stateSeq(1,ii).z;
    Cest = Zest;
    cluster_labels = 1:length(groups);
    cluster_labels = cluster_labels + Total_feats(end);

    for k=1:length(groups)
        cluster_label = cluster_labels(k);
        group = groups{1,k};
        for kk=1:length(group)            
            Cest(find(Zest == group(kk))) = cluster_label;
        end               
    end
    bestGauPsi.stateSeq(1,ii).c = Cest;
end

% %%%% Find Translation and Transformation between similar 


% %%%%%%%%%% Evaluate Results %%%%%%%%%%%%
fprintf('<<<<<<<----------------------------->>>>>>>>>>>\n');
Est_feat_labels = [];
Est_clust_labels = [];

for ii=1:data.N       
    Est_feat_labels = [ Est_feat_labels bestGauPsi.stateSeq(1,ii).z];
    Est_clust_labels = [ Est_clust_labels bestGauPsi.stateSeq(1,ii).c];       
end

% If true labels are given use the external cluster validity indices:
% C1: AR (Adjusted Random Index), C2: Rand Index,  C3: Mirkin Index and C4: Hubert Index 
if ~isempty(data.zTrueAll)
    True_feat_labels = [];
    True_clust_labels = [];
    for ii=1:data.N
    
        True_feat_labels = [ True_feat_labels TruePsi.stateSeq(1,ii).z];
        True_clust_labels = [ True_clust_labels TruePsi.stateSeq(1,ii).c];

    end    
    
    External_indices = [];
    % Estimated feature labels vs true feature labels (BP-HMM vs True
    % unclustered Feats)    
    [AR, Rand, Mirkin, Hubert] = valid_RandIndex(Est_feat_labels,True_feat_labels);
    External_indices (1,:) = [AR, Rand, Mirkin, Hubert];
    fprintf('Estimated BP-HMM feature labels vs True Feature labels: \n');
    fprintf('ARI: %f  Rand: %f Mirkin: %f Hubert: %f\n', AR, Rand, Mirkin, Hubert);   
    
    % Estimated feature labels vs true cluster labels (BP-HMM vs True clusters)
    [AR, Rand, Mirkin, Hubert] = valid_RandIndex(Est_feat_labels,True_clust_labels);
    External_indices (2,:) = [AR, Rand, Mirkin, Hubert];
    fprintf('Estimated BP-HMM feature labels vs True Cluster labels: \n');
    fprintf('ARI: %f  Rand: %f Mirkin: %f Hubert: %f\n', AR, Rand, Mirkin, Hubert);    
    
    % Estimated feature labels vs true cluster labels (BP-HMM vs True clusters)
    [AR, Rand, Mirkin, Hubert] = valid_RandIndex(Est_clust_labels,True_clust_labels);
    External_indices (3,:) = [AR, Rand, Mirkin, Hubert];
    fprintf('Estimated tBP-HMM cluster labels vs True Cluster labels: \n');
    fprintf('ARI: %f  Rand: %f Mirkin: %f Hubert: %f\n', AR, Rand, Mirkin, Hubert);    
end

