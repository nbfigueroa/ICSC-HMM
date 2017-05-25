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


algP = {'Niter', 5,'doSampleFUnique', 0, 'doSplitMerge', 1};  %SM works
% algP = {'Niter', 50,'doSampleFUnique', 1, 'doSampleUniqueZ', 0, 'doSplitMerge', 1, 'RJ.birthPropDistr', 'DataDriven'};   %SM+DD

jobIDGau = ceil(rand*1000);
jobIDAR = ceil(rand*1000);
GauCH = {};

ii = 1;
% for ii=1:1
% Start out with just one feature for all objects
initP  = {'F.nTotal', 1}; 
GauCH{ii} = runBPHMM( data, modelP, {jobIDGau, ii}, algP, initP);
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

if length(unique(labels_aff))==1                    
    labels = labels_kkmeans;                                            
else
    labels = labels_aff;
end
    
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

% %%%% Find Translation and Transformation between similar Emission Models
clc
% close all
dis_ell = 0;
t = {};

for g=1:length(groups)
   
    
    group = groups{g}';
    
    anchor = randsample(group,1);
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
        fprintf('<<<------Estimating Transformation between Theta_%d and Theta_%d--->>>\n', anchor, feature);  
        fprintf('Similarity Probability: %f \n',spcm(anchor, feature, 2))

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
        
        % %%%%% Transform Theta_j to Theta_i %%%%%
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


        % %%%%% Computing Transformation error and proposal distribution %%%%%
        Est_err = EuclideanNormMat(Sigma_i - Sigmaj_i);
        Q_s = exp(Est_err);

        fprintf('--->> Estimation error: %f and Proposal Distribution: %f\n', Est_err, Q_s);

        % %%%%% Display Transformations %%%%%
        if dis_ell
            plotPair3DGauss(Theta_i,Theta_j_i)        
        end 
    
    end    
end
%% %%%%%%%%%% Evaluate Results %%%%%%%%%%%%
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



%% %%%% Find Translation and Transformation between similar Emission Models
close all
dis_ell = 1;

for g=1:length(groups)
    anchor = groups{g}'
    
    hom_ratio = spcm(anchor(1), anchor(2),3);
    homo_dir = spcm(anchor(1), anchor(2),4);
    
    Sigma_a = rec_sigmas{anchor(1)};
    Sigma_b = rec_sigmas{anchor(2)};
    [Va Da] = eig(Sigma_a);
    [Vb Db] = eig(Sigma_b);
  

    if homo_dir < 0
        homo_ratio = hom_ratio;    
    else
        homo_ratio = 1/hom_ratio;
    end
    
    Sigma_b_sc = Vb*(Db*homo_ratio)*inv(Vb);


    % Compute translation
    mu1 = rec_thetas(anchor(1)).mu;
    mu2 = rec_thetas(anchor(2)).mu;
    b = -mu2 + mu1;
    mu2_1 = mu2 + b;

    % Display Transformations
    if dis_ell
        figure('Color', [1 1 1])
        [x,y,z] = created3DgaussianEllipsoid(mu1,Va,Da^1/2);
        mesh(x,y,z,'EdgeColor','black','Edgealpha',0.8);
        hold on
        hidden off

        [Vb_sc Db_sc] = eig(Sigma_b_sc);
        [x,y,z] = created3DgaussianEllipsoid(mu2,Vb_sc,Db_sc^1/2);
        mesh(x,y,z,'EdgeColor','red','Edgealpha',0.8);
        hidden off
        pause(0.1);
        axis equal
    end 

    Mu = [];
    invSigma = [];

    % Find transformation R that minimizes objective function J
    conv_thres = 1e-7;
    thres = 1e-3;
    max_iter = 10000;
    iter = 1;
    J_S = 1;

    tic
    S1 = Sigma_a;
    W2 = eye(size(Sigma_a));
    J = [];
    conv_flag = 0;

    while(J_S > thres)     
        S1 = Sigma_a;
        S2 = W2*Sigma_b_sc*W2';  

        % Objective function Procrustes metric dS(S1,S2) = inf(R)||L1 - L2R||            
        if det(S1)<0
            disp('S1 not pos def.')        
        end

        if det(S2)<0
            disp('S2 not pos def.')
        end

        L1 = matrixSquareRoot(S1);
        L2 = matrixSquareRoot(S2);
        [U,D,V] = svd(L1'*L2);
        R_hat = V*U';    
        J_S = EuclideanNormMat(L1 - L2*R_hat);
        J(iter) = J_S;        

    %     if (mod(iter,10)==0)
    %         if dis_ell 
    %             [V2 D2] = eig(S2);
    %             [x,y,z] = created3DgaussianEllipsoid(mu2_1,V2,D2^1/2);
    %             mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
    %             hidden off   
    %             pause(0.5);
    %         end
    %         fprintf('Error: %f \n',J_S)
    %     end

        %Check convergence
        if (iter>100) && (J(iter-10) - J(iter) < conv_thres)
            disp('Parameter Estimation Converged');
            conv_flag = 1;
            break;
        end

        if (iter>max_iter)
            disp('Exceeded Maximum Iterations');               
            break;
        end

        % Compute approx rotation
        W2 = W2 * R_hat^-1;
        iter = iter + 1;            
    end
    toc

    Mu_1 = rec_thetas(anchor(1)).mu;
    Sigma_1 = rec_thetas(anchor(1)).invSigma^-1;
    [V1 D1] = eig(Sigma_1);

    Mu2_1 = rec_thetas(anchor(2)).mu + b;
    Sigma_2 = rec_thetas(anchor(2)).invSigma^-1;
    [V2 D2] = eig(Sigma_2);
    Sigma_2_sc = V2*(D2*homo_ratio)*inv(V2);
    Sigma2_1 = W2*Sigma_2_sc*W2';

    Trans_err = EuclideanNormMat(Sigma_1 - Sigma2_1)
    Q_s = exp(Trans_err)

    if dis_ell
        figure('Color', [1 1 1])
        [x,y,z] = created3DgaussianEllipsoid(Mu_1,V1,D1^1/2);
        mesh(x,y,z,'EdgeColor','black','Edgealpha',0.8);
        hold on
        hidden off

        [V2_1 D2_1] = eig(Sigma2_1);
        [x,y,z] = created3DgaussianEllipsoid(Mu2_1,V2_1,D2_1^1/2);
        mesh(x,y,z,'EdgeColor','red','Edgealpha',0.8);
        hidden off
        pause(0.1);
        axis equal
    end
end
