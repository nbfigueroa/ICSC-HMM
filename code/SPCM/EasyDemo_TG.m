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
[data, TruePsi] = genToySeqData_Gaussian( 4, 2, 5, 500, 0.5 ); 
% [data, TruePsi] = genToySeqData_Gaussian_null( 5, 2, 5, 500, 0.5 ); 
Robotdata = data;
%%
% Visualize the raw data time series my style
%   with background colored by "true" hidden state
figure( 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5], 'Color',[1 1 1] );
data = Robotdata;
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
close all
% data = Robotdata;

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
            sigmas{i} = TruePsi.theta(i).invSigma^-1;
        end
        
        tau = 5;
        spcm = ComputeSPCMfunctionProb(sigmas, tau);  
        D = log(spcm(:,:,2));        
                
        
        % Compute conditional probabilities p(j|i) proportional to
        % similarity of objects       
        perplexity = 2;
        P_ = d2p(-D,perplexity);
        % Check for NaNs
        nan_ids = find(isnan(P_));
        P_(nan_ids)  = 0;

        % Compute joint probabilities pij (i.e symmetric p(j|i))
        n = size(P_, 1);                                     % number of instances
        P = P_;
        P(1:n + 1:end) = 0;                                 % set diagonal to zero
        P = 0.5 * (P + P');                                 % symmetrize P-values

        figure('Color',[1 1 1], 'Position',[ 3283  545  377 549])
        subplot(2,1,1)
        imagesc(P)
        title('Similarity Probabilities')
        colormap(pink)
        colorbar 

        subplot(2,1,2)        
        yData = tsne_p(P);
        for i=1:size(yData,1)
            scatter(yData(i,1),yData(i,2), 100, [rand rand rand], 'filled')
            hold on
            text(yData(i,1),yData(i,2),num2str(i),'FontSize',20)
        end
        title('Models Respresented in 2-d t-SNE space')
        
        % %%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
        figure('Color', [1 1 1], 'Position',[ 3283  545  377 549]);                   
        subplot(4,1,1)
        imagesc(spcm(:,:,2))
        title('Similarity Function Confusion Matrix')
        colormap(pink)
        colorbar 

        % % %%% Use Kernel-K-means to find the number of clusters from Similarity function  %%%%%%
        N_runs = 10;
        D_kkmeans = P + eye(size(P));
        
        [labels_kkmeans energy] = kernel_kmeans(D_kkmeans, N_runs);
        K = length(unique(labels_kkmeans));

        fprintf('After SPCM and KK-means--->>> \n Number of clusters: %d with total energy %d\n', K, energy);
        subplot(4,1,2)
        imagesc(labels_kkmeans)
        title('Clustering from kernel-kmeans on SPCM Prob. function')
        axis equal tight
        colormap(pink)

        % %%% Compute clusters from Similarity Matrix using Affinity Propagation %%%%%%
        subplot(4,1,3)
        D_aff = diag(median(P,2)) + P;
        [E K labels_aff idx] = affinitypropagation(D_aff);
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
                        
        
        % %%% Compute clusters from Similarity Matrix using Chinese Restaurant Process %%%%%%       
        S = spcm(:,:,2); % Bounded SPCM Similarity Matrix
        M = 2;           % Dimensionality of Spectral Space        
        X = SpectralDimensionalityReduction(S,M);                
        
        % Apply K-means on M-dimensional spectral space
        k = 2;
        opts = statset('Display','final');
        [idx,ctrs] = kmeans(X',k,'Distance','city', 'Replicates', 5,'Options',opts);        
        subplot(4,1,4)
        labels_speckmeans = idx';
        imagesc(labels_speckmeans)
        title('Clustering from kmeans on M-dim Spectral Space of SPCM function')
        axis equal tight
        colormap(pink)
         
%         cluster_labels = [1 2 1 2];
%         clusters = unique(cluster_labels);        
%         figure('Color', [1 1 1])
%         for ii=1:length(clusters)            
%             cluster_ids = find(cluster_labels==clusters(ii))';
%             if M == 2
%                 plot(X(1,cluster_ids)',X(2,cluster_ids)','o','MarkerSize',10, 'MarkerFaceColor', [rand rand rand])
%             end
%             if M == 3
%                 plot3(X(1,cluster_ids)',X(2,cluster_ids)',X(3,cluster_ids)','o','MarkerSize',10, 'MarkerFaceColor', [rand rand rand])
%             end
%             hold on
%         end
%         grid on
%         figure('Color',[1 1 1])        
%         for jj=1:2
%             plot(X(1,idx==jj),X(2,idx==jj),'.', 'Color',[rand rand rand],'MarkerSize',12)
%             hold on            
%         end
%         plot(ctrs(:,1),ctrs(:,2),'ko', 'MarkerSize',12,'LineWidth',2)
%         hold on
%         legend('Cluster 1','Cluster 2','Centroids', 'Location','NW')
%         grid on
        
        
   
        % %%%%%% Generate GROUND TRUTH feature grouping %%%%%%%%%%%%%%
        groups = {};          
        for c=1:K; groups{1,c} = find(labels==c); end
        
        
        % %%%%%% Visualize GROUND TRUTH segmentation, features and groupings %%%%%%%%%%%%%%
        [Segm_results Total_feats hIM my_color_map] = plotSegDataNadia(data, TruePsi, 1:data.N, 'Real state sequence', groups);
        
        
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
        [Segm_results Total_feats hIM my_color_map] = plotSegDataNadia(data, TruePsi, 1:data.N, 'Real state sequence', []);
    
    end
end

%%
% -------------------------------------------------   RUN MCMC INFERENCE!
clc
close all
data = Robotdata;
% addpath(genpath('/home/nadiafigueroa/dev/MATLAB/tGau-BP-HMM'))
% cd('/home/nadiafigueroa/dev/MATLAB/tGau-BP-HMM')
% modelP = {'bpM.gamma', 2}; 
% algP   = {'Niter', 100, 'HMM.doSampleHypers',0,'BP.doSampleMass',0,'BP.doSampleConc',0}; 
% algP = {'Niter', 1,'doSampleFUnique', 1, 'doSplitMerge', 1};

% TESTS (Jan 12, 2014)
modelP = {'bpM.gamma', 3,'obsM.Scoef',1,'bpM.c', 1}; %for rotation invariance (loose)

% For hesitation data testing
% modelP = {'bpM.gamma', 3,'obsM.Scoef',1,'bpM.c', 1,'hmmM.alpha' , 2,'hmmM.kappa',2000};


algP = {'Niter', 300,'doSampleFUnique', 0, 'doSplitMerge', 1};  %SM works
% algP = {'Niter', 50,'doSampleFUnique',1, 'doSampleUniqueZ', 0, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven'};   %SM+DD

jobIDGau = ceil(rand*1000);
jobIDAR = ceil(rand*1000);
GauCH = {};


ii = 1;
for ii=1:1
% Start out with just one feature for all objects
initP  = {'F.nTotal', 1}; 
GauCH{ii} = runBPHMM( data, modelP, {jobIDGau, ii}, algP, initP);
end

%% VISUALIZE SEGMENTATION RESULTS FOR CLUSTERING W/GAUSSIAN
clc
close all
clear bestGauCH
clear bestGauPsi

cd('/home/nadiafigueroa/dev/MATLAB/')
addpath(genpath('/home/nadiafigueroa/dev/MATLAB/tGau-BP-HMM'))


[bestGauCH] = getresults(GauCH,1);
fprintf('Model Type: Multivariate Gaussian\n');
fprintf( '---Best trial "Chain History"--- \n');
fprintf('Trial: %d Iteration: %d\n',bestGauCH.trial, bestGauCH.iter)
bestGauPsi = bestGauCH.Psi;ls

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

sigmas = rec_sigmas;

%% %%%%% Visualize Prob. Similarity Confusion Matrix %%%%%%%%%%%%%%
close all
% tau = 1;
tau = 5;
spcm = ComputeSPCMfunctionProb(rec_sigmas, tau);  
D = log(spcm(:,:,2));

% Compute conditional probabilities p(j|i) proportional to
% similarity of objects       
perplexity = 1; % for 2d data
perplexity = 2; % for n-d data
P_ = d2p(-D,perplexity);
% Check for NaNs
nan_ids = find(isnan(P_));
P_(nan_ids)  = 0;

% Compute joint probabilities pij (i.e symmetric p(j|i))
n = size(P_, 1);                                     % number of instances
P = P_;
P(1:n + 1:end) = 0;                                 % set diagonal to zero
P = 0.5 * (P + P');                                 % symmetrize P-values

figure('Color',[1 1 1], 'Position',[ 3283  545  377 549])
subplot(2,1,1)
imagesc(P)
title('Similarity Probabilities')
colormap(pink)
colorbar 

subplot(2,1,2)
yData = tsne_p(P);
for i=1:size(yData,1)
scatter(yData(i,1),yData(i,2), 100, [rand rand rand], 'filled')
hold on
text(yData(i,1),yData(i,2),num2str(i),'FontSize',20)
end
title('Models Respresented in 2-d t-SNE space')


figure('Color', [1 1 1], 'Position',[ 3283  545  377 549]);           
subplot(4,1,1)
imagesc(exp(D))
title('Similarity function Kernel Matrix')
colormap(pink)
colorbar 

%%% Use Kernel-K-means to find the number of clusters from Similarity function  %%%%%%
N_runs = 10;
D_kkmeans = P + eye(size(P));        
[labels_kkmeans energy] = kernel_kmeans(D_kkmeans, N_runs);
K = length(unique(labels_kkmeans));
fprintf('After SPCM and KK-means--->>> \n Number of clusters: %d with total energy %d\n', K, energy);
subplot(4,1,2)
imagesc(labels_kkmeans)
title('Clustering from kernel-kmeans on SPCM Prob. function')
axis equal tight
colormap(pink)
% energy = 0;

% %%% Compute clusters from Similarity Matrix using Affinity Propagation %%%%%%
subplot(4,1,3)
D_aff = diag(mean(P,2)) + P;
[E K_ labels_aff idx] = affinitypropagation(D_aff);
fprintf('After Affine Propagation on SPCM Prob. function--->>> \n Number of clusters: %d\n', K_);
imagesc(labels_aff')
title('Clustering from Aff. Prop. on SPCM Prob. function')
axis equal tight
colormap(pink)

labels = labels_aff;

if length(unique(labels_aff))==1                                       
    display('Chose kkmeans');
    labels = labels_kkmeans;                                            
else
    if energy < 2e-7
        if length(unique(labels_aff))==length(labels_aff)
            display('Chose kkmeans');
            labels = labels_kkmeans;        
        else
            display('Chose aff');
            labels = labels_aff;
            K = K_;
        end
    else
        display('Chose aff');
        labels = labels_aff;
        K = K_;
    end
end


% %%% Compute clusters from Similarity Matrix using Chinese Restaurant Process %%%%%%       
S = spcm(:,:,2); % Bounded SPCM Similarity Matrix
M = 2;           % Dimensionality of Spectral Space        
X = SpectralDimensionalityReduction(S,M);                

% Apply K-means on M-dimensional spectral space
k = 2;
opts = statset('Display','final');
[idx,ctrs] = kmeans(X',k,'Distance','city', 'Replicates', 5,'Options',opts);        
subplot(4,1,4)
labels_SpecKMeans = idx';
imagesc(labels_SpecKMeans)
title('Clustering from kmeans on M-dim Spectral Space of SPCM function')
axis equal tight
colormap(pink)
        
%% %%%%%% Generate Estimated feature grouping and visualize segmentation %%%%%%%%%%%%%%
groups = {};          
for c=1:K; groups{1,c} = find(labels==c); end

% Toy Dataset
[Segm_results Total_feats hIM my_color_map] = plotSegDataNadia(data, bestGauPsi, 1:data.N, 'Best recovered state sequence',groups);

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

%%%%%%%%% Extract Segmented trajectories%%%%%%%%%
clear Feat_data
clear Feat_data_ori
% Adjust Segm_results for toy data
toy_data = 1;
if toy_data
    Segm_results_adj = Segm_results;
    T = 0;
    for i=1:length(Segm_results)
        tmp = Segm_results{i};
        tmp = [tmp(:,1) tmp(:,2)+T];
        Segm_results_adj{i} = tmp;
        T = T + 500;
    end

    SegmentedData = {};
    for i=1:length(Total_feats)
        l = 1;
        Xdata_seg = [];
        for j=1:length(Segm_results)
            segments = Segm_results_adj{j};
            seg_id = find(segments(:,1)==Total_feats(i));

            if ~isempty(seg_id)            
              for k=1:length(seg_id)
                  if seg_id(k)==1
                      if j == 1
                        start_seg = 1;
                      else
                        start_seg = (j-1)*500;
                      end
                  else
                      start_seg = segments(seg_id(k)-1,2);
                  end
                  end_seg = segments(seg_id(k),2);

                  tmp_data.ts = j;
                  tmp_data.start_seg = start_seg;
                  tmp_data.end_seg = end_seg;
                  tmp_data.raw = data.Xdata(:,start_seg:end_seg);
                  SegmentedData{i,l} = tmp_data;

                  Xdata_seg = [Xdata_seg tmp_data.raw];
                  l = l+1;
              end 
            end
        end
         Feat_data{i,1} = Xdata_seg;
    end
else
    
%     X = data.seq(ii);
    
   SegmentedData = {};
    for i=1:length(Total_feats)
        l = 1;
        Xdata_seg = [];
        Xdata_seg_ori = [];
        
        for j=1:length(Segm_results)
            X_j = data.seq(j);
            X_j_ori = XnT{j};
            segments = Segm_results{j};
            seg_id = find(segments(:,1)==Total_feats(i));

            if ~isempty(seg_id)            
              for k=1:length(seg_id)                  
                  if seg_id(k)==1                      
                      start_seg = 1;                      
                  else
                      start_seg = segments(seg_id(k)-1,2);
                  end
                  end_seg = segments(seg_id(k),2);

                  tmp_data.ts = j;
                  tmp_data.start_seg = start_seg;
                  tmp_data.end_seg = end_seg;
                  tmp_data.raw  = X_j(:,start_seg:end_seg);
                  tmp_data.orig = X_j_ori(:,start_seg:end_seg);
                  SegmentedData{i,l} = tmp_data;

                  Xdata_seg     = [Xdata_seg tmp_data.raw];
                  Xdata_seg_ori = [Xdata_seg_ori tmp_data.orig];
                  l = l+1;
              end 
            end
        end
         Feat_data{i,1} = Xdata_seg;
         Feat_data_ori{i,1} = Xdata_seg_ori;
         
    end
    
    
    %%% Plot 3d trajectories of extracted features
    figure('Color',[1 1 1])
%     for f=1:length(Feat_data)
    for f=1:2
        feat_data = Feat_data_ori{f};
        scatter3(feat_data(1,:),feat_data(2,:),feat_data(3,:),5, my_color_map(f,:),'filled')
        hold on
    end
   grid on
   axis ij
   xlabel('x')
   ylabel('y')
   zlabel('z')
        
end

%% %%%% Find Translation and Transformation between similar Emission Models
clc
% close all
dis_ell = 0;
t = {};

for g=1:length(groups)
       
    group = groups{g}';    
    rand_id = randi(length(group),1);
    anchor = group(rand_id);
    anchor_idx = find(group==anchor);
    feats = group;
    feats(anchor_idx) = [];
    
    dim = length(rec_thetas(anchor).mu);
    
    t{g,1}.feat = anchor;
    t{g,1}.b = zeros(dim, 1);
    t{g,1}.h = 1;
    t{g,1}.W = eye(dim);
    t{g,1}.J = [];        
    
    for f=1:length(feats)

        
        feature = feats(f);               
        prob_sim = max(P(feature,anchor),P(anchor,feature));
        
        fprintf('<<<------Estimating Transformation between Theta_%d and Theta_%d (sim prob: %f)--->>>\n', feature, anchor, prob_sim);  

        hom_ratio = spcm(anchor, feature, 3);
        hom_dir = spcm(anchor, feature,4);

        Theta_i = rec_thetas(anchor);
        Theta_j = rec_thetas(feature);

        % %%%%% Display Transformations %%%%%
        if dis_ell
            plotPair3DGauss(Theta_i,Theta_j)
        end 

        % %%%%% Computing translation, scale and Transformation between 2
        % gaussians %%%%%
        % Transformation is from \Theta_j to \Theta_i
        [W b h J] = computeGaussTransformation(Theta_i,Theta_j, hom_ratio, hom_dir);
        
        
        t{g,f+1}.feat = feature;
        t{g,f+1}.b = b;
        t{g,f+1}.h = h;
        t{g,f+1}.W = W;
        t{g,f+1}.J = J;
        t{g,f+1}.prob_sim = prob_sim;
        
        % %%%%% Transform Theta_i to Theta_j %%%%%
        Mu_i = Theta_i.mu;
        Sigma_i = Theta_i.invSigma^-1;

        % Translate Mu2 to Mu2_1
        Muj_i = Theta_j.mu + b;

        % Scale Sigma 2 by h
        Sigma_j = Theta_j.invSigma^-1;
        [Vj Dj] = eig(Sigma_j);
        Sigma_j_sc = Vj*(h*Dj)*inv(Vj);

        % Transform Sigma2 by WSigmaW'
        Sigmaj_i = W*Sigma_j_sc*W';

        Theta_j_i.mu = Muj_i;
        Theta_j_i.invSigma = Sigmaj_i^-1;


        % %%%%% Computing Transformation error and Probability of Transformation %%%%%
        Est_err = EuclideanNormMat(Sigma_i - Sigmaj_i);
        alpha = size(Sigma_j,1);
        p_trans = exp(-1/alpha*Est_err);

        t{g,f+1}.prob_trans = p_trans;
        
        fprintf('--->> Estimation error: %f and PI_T(Theta_%d->Theta_%d), trans_prob: %f\n', Est_err, feature, anchor,  p_trans);

        % %%%%% Display Transformations %%%%%
        if dis_ell
            plotPair3DGauss(Theta_i,Theta_j_i)        
        end 
    
    end    
end
bestGauPsi.t = t;

%%%%%% Compute marginal likelihood probabilities of extracted data/parameters from original BPHMM 
% clc
% close all
nb_c = size(t,1);
bestGauPsi.theta_groups = groups;
trans_theta = bestGauPsi.theta;
logSoftEv = [];
margLogPr = [];
n_data = [];
l_probs_t = [];
% Joint log probability of current feature partition
for i=1:size(bestGauPsi.theta,2)
    theta = bestGauPsi.theta(i);
    feat_data = Feat_data{i};
    n_data = [n_data; size(feat_data,2)];
    logSoftEv = calcLogSoftEv(theta, feat_data);
    l_prob_t = -20;
    margLogPr(i,1) =  calcLogMargPrObsSeqFAST(logSoftEv',1) + l_prob_t;    
end
N = sum(n_data);
w = n_data/N;
% margLogPr_joint = sum(w.*margLogPr);
margLogPr_joint = sum(margLogPr);

%%%%%% Compute marginal likelihood probabilities of extracted transformed data/parameters from tGau-BPHMM 
% Joint log probability of proposed feature partition
margLogPr_prop = zeros(size(margLogPr));
n_data = zeros(size(margLogPr));
trans_data = [];
trans_theta  = bestGauPsi.theta;
bestGauPsiTrans = bestGauPsi;
l_probs_t = [];
anchor_ids = [];

for i=1:size(t,1)
    anchor_data = [];
    for j=1:size(t,2)      
         if isfield(t{i,j},'feat')
            feat_id = t{i,j}.feat;        
            theta = bestGauPsi.theta(feat_id);

            feat_data = Feat_data{feat_id};
            n_data(feat_id,1) = size(feat_data,2);

            if j==1
                % This is an anchor feature so no transform            
                trans_data{feat_id}  = Feat_data{feat_id};            
                anchor_data = Feat_data{feat_id};
                anchor_id = feat_id;  
                trans_theta(feat_id) = theta;
                                    
                logSoftEv = calcLogSoftEv(theta, feat_data);
                margLogPr_prop(feat_id, 1) = calcLogMargPrObsSeqFAST(logSoftEv',1);
                anchor_ids(feat_id, 1) = feat_id; 
            else            
                % Transform anchor theta to feature                    
                l_prob_sim = log(t{i,j}.prob_sim);
                l_prob_trans = log(t{i,j}.prob_trans);
                l_prob_t = l_prob_sim  + l_prob_trans; 

                % Extract transform parametrs            
                b = t{i,j}.b;
                h = t{i,j}.h;
                W = t{i,j}.W;

                tmp = bsxfun(@minus, feat_data, theta.mu )*h;
                transformed_data = tmp;
                for m=1:size(feat_data,2)
                    transformed_data(:,m) = (tmp(:,m)'*W^-1)';
                end
                transformed_data = bsxfun(@plus, transformed_data, b + theta.mu );
                trans_data{feat_id}  = transformed_data;     

                % transform theta
                t_theta = bestGauPsi.theta(feat_id);
                t_theta.mu = t_theta.mu + b;
                sigma = t_theta.invSigma^-1;
                [V D] = eig(sigma);
                sigma_sc = V*(h*D)*inv(V);
                sigma_t = W*sigma_sc*W';
                t_theta.invSigma = sigma_t^-1;

                trans_theta(feat_id) = t_theta;

                theta = t_theta;
                theta = trans_theta(anchor_id);
                feat_data = transformed_data;
                
                
                
                logSoftEv = calcLogSoftEv(theta, feat_data);
                margLogPr_prop(feat_id, 1) = calcLogMargPrObsSeqFAST(logSoftEv',1) + margLogPr_prop(anchor_id,1) + l_prob_t;
                anchor_ids(feat_id, 1) = anchor_id; 
            end            
        end
    end
    
    if (margLogPr_prop(feat_id, 1) > 1000)
        % if logPr of data is too positive give high positive value to anchor
        margLogPr_prop(anchor_id,1) = 200000;
    else
        % if logPr of data is too negative give low negative value to anchor
        margLogPr_prop(anchor_id,1) = -20;
    end
end

bestGauPsiTrans.theta = trans_theta;
margLogPr_prop_joint = sum(margLogPr_prop);

margLogPr_prop_joint
margLogPr_joint

% -------------------------------- Accept or Reject!
logPrAccept = margLogPr_prop_joint - margLogPr_joint;
rho = exp( logPrAccept )
rho = min(1, rho);

doAccept = rand < rho;

if doAccept
    fprintf('Accept partition with logProb %f \n', margLogPr_prop_joint);    


    if toy_data
        figure('Color', [1 1 1], 'Position',[1383 320 537 774])
        subplot(2,1,1)
        for i=1:length(Total_feats)
            scatter(Feat_data{i}(1,:),Feat_data{i}(2,:),50,my_color_map(i,:))
            hold on
        end
        plotEmissionParams(bestGauPsi,my_color_map)
        title('Data and Recovered Emission Parameters')


        subplot(2,1,2)
        for i=1:length(Total_feats)
            scatter(trans_data{i}(1,:),trans_data{i}(2,:),50,my_color_map(i,:))
            hold on
        end
        plotEmissionParams(bestGauPsiTrans,my_color_map)
        title('Transformed Data and Grouped Emission Parameters')
    else
        %     close
        %%% Plot 3d trajectories of extracted features
        figure('Color',[1 1 1])
        for f=1:length(Feat_data)
            feat_data = Feat_data_ori{f};
            anchor_id = anchor_ids(f,1)+2;
            scatter3(feat_data(1,:),feat_data(2,:),feat_data(3,:),5, my_color_map(anchor_id,:),'filled')
            hold on
        end
        grid on
        axis ij
        xlabel('x')
        ylabel('y')
        zlabel('z')
    end


else
    fprintf('Keep current feature partition \n');
end


