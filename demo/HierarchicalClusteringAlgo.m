%% Underlying Gaussian Computation w/toy gaussians
clear all
figure('Color',[1 1 1]);
angle = pi/2;
%Ellipsoid 1
Cov = ones(3,3) + diag([1 1 1]);
% mu = [1 1 2]';
mu = [0 0 0]';
behavs_theta{1,1} = Cov;
[V1,D1] = eig(Cov);
% CoordRot = rotx(angle/3)*roty(angle/3)*rotz(angle/3);
CoordRot = rotx(-angle);
% CoordRot = eye(3);
% V1_rot = V1* CoordRot;
% Covr1 = V1_rot*D1*inv(V1_rot);
% [V1,D1] = eig(Covr1);
[x,y,z] = created3DgaussianEllipsoid(mu,V1,D1^1/2);
mesh(x,y,z,'EdgeColor','blue','Edgealpha',0.2);
hidden off
hold on;

%Ellipsoid 2: Scale+Noise
D1m = diag(D1)*1.3 + abs(randn(3,1).*[0.35 0.37 0.3]');
D1m = diag(D1)*0.5;
Covs2 = V1*(diag(D1m))*V1';
behavs_theta{1,2} = Covs2;
[V2,D2] = eig(Covs2);
[x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
mesh(x,y,z,'EdgeColor','black','Edgealpha',0.2);
hidden off
hold on;


%Ellipsoid 3: Rotated Coordinates
% CoordRot = rotx(angle)*roty(angle*0.5)*rotx(angle*2);
CoordRot = [1 0 0; 0 1 0 ; 0 0 -1]
% CoordRot = eye(3);
% CoordRot = rotx(angle);
V2_rot = CoordRot*V2;
Covs3 = V2_rot*D2*V2_rot';
behavs_theta{1,3} = Covs3;
[V3,D3] = eig(Covs3);
mu = [0 0 0]';
[x,y,z] = created3DgaussianEllipsoid(mu,V3,D3^1/2);
mesh(x,y,z,'EdgeColor','red','Edgealpha',0.2);
hidden off
hold on;

colormap jet
alpha(0.5)
% xlabel('x');ylabel('y');zlabel('z');
% grid off
% axis off
axis equal



%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrix Similarity on Grasps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

behavs = [];
behavs_theta = [];
load 6D-Grasps.mat
dim = 6; 
for i=1:size(behavs_theta,1)
    behavs{i,1} = [1:size(behavs_theta,2)] + (i-1)*size(behavs_theta,2);
end

% Prepare Data for Full Comparison
Sigmas =[];
Eig_behavs = [];
Ids_behavs = [];
k=1;
for i=1:size(behavs_theta,1)
    for j=1:size(behavs_theta,2)
        if ~isempty(behavs_theta{i,j})
            %Real Sigmas
            Sigmas{k}= behavs_theta{i,j}.Sigma(1:dim,1:dim);
            
            %Simulated Sigmas
            [V,D] = eig(Sigmas{k});
                      
            %Behavs ID on behavs_theta
            Ids_behavs{k} = [i j]; 
            
            %Eigen Stuff
            eigstuff.V = V;
            eigstuff.D = D;
            Eig_behavs{k} = eigstuff;
            k = k+1;            
        end
    end
end

%% Compute SPCM function Confusion Matrix
spcm = [];
% for i=1:length(Eig_behavs)
%   for j=1:length(Eig_behavs)     
       
for i=1:length(behavs_theta)
  for j=1:length(behavs_theta)     
      
         % Testing w/ Example Ellipsoids
        [Vi, Di] = eig(behavs_theta{i});
        [Vj, Dj] = eig(behavs_theta{j});
        
        %For Datasets
%         Vi = Eig_behavs{i}.V; Di = Eig_behavs{i}.D;        
%         Vj = Eig_behavs{i}.V; Dj = Eig_behavs{j}.D;

        %Ensure eigenvalues are sorted in ascending order
        [Vi, Di] = sortem(Vi,Di);
        [Vj, Dj] = sortem(Vj,Dj);
        
        %Structural of Sprectral Polytope
        Xi = Vi*Di^1/2;
        Xj = Vj*Dj^1/2;
                
        %Norms of Spectral Polytope Vectors
        for k=1:length(Dj)
            eig_i(k,1) = norm(Xi(:,k));
            eig_j(k,1) = norm(Xj(:,k));
        end
        
        %Homothetic factors
        hom_fact_ij = eig_i./eig_j;
        hom_fact_ji = eig_j./eig_i;
        
        %Magnif
        if (mean(hom_fact_ji) > mean(hom_fact_ij)) || (mean(hom_fact_ji) == mean(hom_fact_ij))
            dir = 1;
            hom_fact = hom_fact_ji;
        else
            dir = -1;
            hom_fact = hom_fact_ij;
        end     
        
        spcm(i,j,1) = std(hom_fact); 
        spcm(i,j,2) = mean(hom_fact); 
        spcm(i,j,3) = dir; 
        
   end
end

figure('Color', [1 1 1]);
spcm_cut = spcm(:,:,1);
cut_thres = 100;
for i=1:size(spcm,1)
    for j=1:size(spcm,2)
        if spcm(i,j)>cut_thres
            spcm_cut(i,j) = cut_thres;
        end
    end
end

imagesc(-log(spcm(:,:,1)))
title('log(SPCM) Confusion Matrix')
colormap(copper)
colormap(pink)
colorbar 

    % Variants of Non-Euclidean Statistics for Covariance Matrices
        % Objective function Root Euclidean dH(S1,S2) = ||(S1)^1/2 - (S2)^1/2||
    %     J_H = EuclideanNormMat(S1^1/2 - S2^1/2)
        % Objective function Cholesky dC(S1,S2) = ||chol(S1) - chol(S2)||
    %     J_C = EuclideanNormMat(chol(S1) - chol(S2))
        % Objective function Cholesky dL(S1,S2) = ||log(S1) - log(S2)||
    %     J_L = EuclideanNormMat(log(S1) - log(S2))   


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Construct Hierarchical Gaussian Models with Tree Struct %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display iterations
dis = 0;
clear Xi
Xi = {}; 

% Create tree
root.name = 'Behaviors';
root.idx = 0;
root.Sigma = eye(size(behavs_theta{1}));

clear t
t = tree(root);
n = [];
for i=1:length(behavs_theta)
    name = strcat('Theta_',num2str(i));
    node.name = name;
    node.idx = i; %bphmm idx
    node.Sigma = behavs_theta{i};
    [t n(i)] = t.addnode(1,node);
end
tnames = t.treefun(@getName);
disp(tnames.tostring)
figure('Color', [1 1 1]);tnames.plot

clear xi
xi.behavs = t; 
xi.node_ids = n;

alpha = 1;
ii = 1;
no_more = 1;
%  while(no_more || length(xi.node_ids)>1)
for ii=1:1

    % Extract Behaviors from 1st level
    sigmas = [];
    % Behavs to compare
    for i=1:length(xi.node_ids)
        sigmas{i} =  xi.behavs.get(xi.node_ids(i)).Sigma;
    end
    
    % Compute Similarity
    spcm = ComputeSPCMfunction(sigmas,1);    
    xi.spcm = spcm; 
    
    % Find Anchor pair
    spcm_pairs = nchoosek(1:1:length(xi.node_ids),2);
    for i=1:size(spcm_pairs,1)
        spcm_pairs(i,3) = spcm(spcm_pairs(i,1),spcm_pairs(i,2),1);
    end
    [min_spcm min_spcm_id] = min(spcm(:,3));
    
    if (min_spcm < alpha)
        anchor = spcm_pairs(min_spcm_id,1:2);

        % Find Anchor, Homothetic Ratio and Directionality
        homo_dir = spcm(anchor(1),anchor(2),3);
        if homo_dir < 0
            anchor = [anchor(2) anchor(1)];
        end    
        homo_ratio = spcm(anchor(1),anchor(2),2); 

        h(1) = 1;
        h(2) = homo_ratio;

        Sigma_a = sigmas{anchor(1)};
        Sigma_b = sigmas{anchor(2)};
        [Va Da] = eig(Sigma_a);
        [Vb Db] = eig(Sigma_b);
        Sigma_b_sc = Vb*(Db*1/h(2))*inv(Vb);


        % Display Transformations
        mu = [0 0 0]';
        if dis
            figure('Color', [1 1 1])
            [x,y,z] = created3DgaussianEllipsoid(mu,Va,Da^1/2);
            mesh(x,y,z,'EdgeColor','black','Edgealpha',0.8);
            hidden off
            hold on;
            axis equal
            [x,y,z] = created3DgaussianEllipsoid(mu,Vb,Db^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
            hidden off   
            pause(1);
            [Vb_sc Db_sc] = eig(Sigma_b_sc);
            [x,y,z] = created3DgaussianEllipsoid(mu,Vb_sc,Db_sc^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
            hidden off
            pause(1);

        end 

        % Find transformation R that minimizes objective function J
        thres = 1e-5;
        max_iter = 1000;
        iter = 1;
        J_S = 1;

        tic
        S1 = Sigma_a;
        W2 = eye(size(Sigma_a));
        W = [];
        while(J_S > thres || iter > max_iter)     
            S1 = Sigma_a;
            S2 = W2*Sigma_b_sc*W2';  

            if dis && (mod(iter,5)==0)
                [V2 D2] = eig(S2);
                [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
                mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
                hidden off   
                pause(1);
            end

            % Objective function Procrustes metric dS(S1,S2) = inf(R)||L1 - L2R||
            L1 = chol(S1);
            L2 = chol(S2);
            [U,D,V] = svd(L1'*L2);
            R_hat = V*U';    
            J_S = EuclideanNormMat(L1 - L2*R_hat);

            % Compute approx rotation
            W2 = W2 * R_hat^-1;
            iter = iter + 1;
        end

        S = [];
        S(:,:,1) = S1;
        S(:,:,2) = S2;


        W(:,:,1) = eye(size(S1));
        W(:,:,2) = W2;

        if dis
            [V2 D2] = eig(S2);
            [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.8);
            hidden off
            axis equal
            pause(1);
        end
        toc

        fprintf('Finished pair %d and %d at iteration %d - Error: %f ',anchor(1), anchor(2), iter,J_S)

        % Least Square Estimator of Average Covariance Matrix 
        % Using the Log-Euclidean Distance (Good for Matrix Interpolation)
        n = size(S,3);
        Delta_hat = zeros(size(S(:,:,1)));
        for i=1:size(S,3)
            Delta_hat =  Delta_hat + log(S(:,:,i));
        end
        Delta_hat = (1/n)*Delta_hat;
        Sigma_hat = real(exp(Delta_hat));

        xi.anchor = xi.node_ids(anchor);
        xi.homo = h;
        xi.W = W;
        xi.S = S;
        xi.Parent_node = Sigma_hat;
        Xi{ii} = xi;

        % Update Tree
        ii = ii + 1;

        %Create New Group Node/tree
        clear t
        t = xi.behavs;
        new_node_idx = t.get(xi.node_ids(end)).idx + 1;
        name = strcat('Theta_',num2str(new_node_idx));
        root.name = name;
        root.idx = new_node_idx; %bphmm idx
        root.Sigma = Sigma_hat;
        ti = tree(root);
        for j=1:2
            node = t.get(xi.anchor(j));
            node.homo = xi.homo(j);
            node.W = xi.W(:,:,j);
            if j == 1
                node.spcm = xi.spcm(anchor(1),anchor(2),:);
            else
                node.spcm = xi.spcm(anchor(2),anchor(1),:);
            end
            ti = ti.addnode(1,node);
        end
        
        chop_nodes = sort(xi.anchor,'descend');
        for j=1:2
            t = t.chop(chop_nodes(j));
        end
        
        t = t.graft(1,ti);        
        tnames = t.treefun(@getName);
        disp(tnames.tostring)
        figure('Color', [1 1 1]);tnames.plot
        
        clear xi
        xi.behavs = t; 
        Xi{ii} = xi;
    else
        disp('No more similar behaviors');
        no_more = 0;
    end
    

end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Construct Hierarchical Gaussian Models %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display iterations
dis = 1;
clear Xi
Xi = {}; 

clear xi
xi.behavs = behavs_theta; 

ii = 1;
 while(length(xi.behavs)>1)
% for ii=1:1

    % Compute Similarity
    spcm = ComputeSPCMfunction(xi.behavs,1);
    xi.spcm = spcm; 
    
    % Find Anchor pair
    spcm_pairs = nchoosek(1:1:length(xi.behavs),2);
    for i=1:size(spcm_pairs,1)
        spcm_pairs(i,3) = spcm(spcm_pairs(i,1),spcm_pairs(i,2),1);
    end
    [min_spcm min_spcm_id] = min(spcm(:,3));
    anchor = spcm_pairs(min_spcm_id,1:2);

    
    % Find Anchor, Homothetic Ratio and Directionality
    homo_dir = spcm(anchor(1),anchor(2),3);
    if homo_dir < 0
        anchor = [anchor(2) anchor(1)];
    end    
    homo_ratio = spcm(anchor(1),anchor(2),2); 

    h(1) = 1;
    h(2) = homo_ratio;

    Sigma_a = behavs_theta{anchor(1)};
    Sigma_b = behavs_theta{anchor(2)};
    [Va Da] = eig(Sigma_a);
    [Vb Db] = eig(Sigma_b);
    Sigma_b_sc = Vb*(Db*1/h(2))*inv(Vb);

 
    % Display Transformations
    mu = [0 0 0]';
    if dis
        figure('Color', [1 1 1])
        [x,y,z] = created3DgaussianEllipsoid(mu,Va,Da^1/2);
        mesh(x,y,z,'EdgeColor','black','Edgealpha',0.8);
        hidden off
        hold on;
        axis equal
        [x,y,z] = created3DgaussianEllipsoid(mu,Vb,Db^1/2);
        mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
        hidden off   
        pause(1);
        [Vb_sc Db_sc] = eig(Sigma_b_sc);
        [x,y,z] = created3DgaussianEllipsoid(mu,Vb_sc,Db_sc^1/2);
        mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
        hidden off
        pause(1);

    end 

    % Find transformation R that minimizes objective function J
    thres = 1e-5;
    max_iter = 1000;
    iter = 1;
    J_S = 1;

    tic
    S1 = Sigma_a;
    W2 = eye(size(Sigma_a));
    W = [];
    while(J_S > thres || iter > max_iter)     
        S1 = Sigma_a;
        S2 = W2*Sigma_b_sc*W2';  

        if dis && (mod(iter,5)==0)
            [V2 D2] = eig(S2);
            [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
            hidden off   
            pause(1);
        end

        % Objective function Procrustes metric dS(S1,S2) = inf(R)||L1 - L2R||
        L1 = chol(S1);
        L2 = chol(S2);
        [U,D,V] = svd(L1'*L2);
        R_hat = V*U';    
        J_S = EuclideanNormMat(L1 - L2*R_hat);

        % Compute approx rotation
        W2 = W2 * R_hat^-1;
        iter = iter + 1;
    end

    S = [];
    S(:,:,1) = S1;
    S(:,:,2) = S2;


    W(:,:,1) = eye(size(S1));
    W(:,:,2) = W2;

    if dis
        [V2 D2] = eig(S2);
        [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
        mesh(x,y,z,'EdgeColor','red','Edgealpha',0.8);
        hidden off
        axis equal
        pause(1);
    end
    toc

    fprintf('Finished pair %d and %d at iteration %d - Error: %f ',anchor(1), anchor(2), iter,J_S)

    % Least Square Estimator of Average Covariance Matrix 
    % Using the Log-Euclidean Distance (Good for Matrix Interpolation)
    n = size(S,3);
    Delta_hat = zeros(size(S(:,:,1)));
    for i=1:size(S,3)
        Delta_hat =  Delta_hat + log(S(:,:,i));
    end
    Delta_hat = (1/n)*Delta_hat;
    Sigma_hat = real(exp(Delta_hat));

    xi.anchor = anchor;
    xi.homo = h;
    xi.W = W;
    xi.S = S;
    xi.Sigma_hat = Sigma_hat;
    Xi{ii} = xi;
    
    new_behavs = xi.behavs;
    new_behavs(anchor) = [];
    new_behavs = [new_behavs Sigma_hat];
    clear xi
    xi.behavs = new_behavs;
    ii=ii+1;
    Xi{ii} = xi;
end

%% Make groups

alpha  = 1.5;
groups = [];
groups = MotionGrouping(spcm(:,:,1),alpha);

% Plot Grasp Ellipsoids
fig_row = floor(sqrt(size(groups,1)));
fig_col = size(groups,1)/fig_row;
figure('Color',[1 1 1])
for ii=1:length(groups)
    g = groups{ii};
    gs = [];
    subplot(fig_row,fig_col,ii)
% figure('Color',[1 1 1])
    mu = [0 0 0]';
    for jj=1:length(g)
        behav.g_id = g(jj);
        behav.bt_id = Ids_behavs{g(jj)};
        behav.seg_id = behavs{behav.bt_id(1)}(behav.bt_id(2));
        behav.mu = behavs_theta{behav.bt_id(1),behav.bt_id(2)}.mu;
        behav.Sigma = behavs_theta{behav.bt_id(1),behav.bt_id(2)}.Sigma;
        behav.V = Eig_behavs{g(jj)}.V;
        behav.D = Eig_behavs{g(jj)}.D;
        
        gs = [gs behav.seg_id];
        grouped_behavs{ii,jj} = behav;
        [x,y,z] = created3DgaussianEllipsoid(mu,behav.V(1:3,1:3),behav.D(1:3,1:3));
        surfl(x,y,z);
%         mu = mu + [0 0.5 0]';
        hold on
    end
    if length(gs)<10
        tit = strcat('Group ', num2str(ii),' --> Behavs: ',num2str(gs));
    else
        tit = strcat('Group ', num2str(ii),' --> Behavs: ',num2str(gs(1)), ' to ', num2str(gs(end)));
    end
    title(tit)
    xlabel('x');ylabel('y');zlabel('z');
    axis equal
end


%% Procrustes
A = [11 39;17 42;25 42;25 40;23 36;19 35;30 34;35 29;...
30 20;18 19];
B = [15 31;20 37;30 40;29 35;25 29;29 31;31 31;35 20;...
29 10;25 18];
 
X = A;
Y = B + repmat([25 0], 10,1); 
figure('Color', [1 1 1])
plot(X(:,1), X(:,2),'r-', Y(:,1), Y(:,2),'b-');
text(X(:,1), X(:,2),('abcdefghij')')
text(Y(:,1), Y(:,2),('abcdefghij')')
legend('X = Target','Y = Comparison','location','SE')
set(gca,'YLim',[0 55],'XLim',[0 65]);

[d, Z, tr] = procrustes(X,Y);
plot(X(:,1), X(:,2),'r-', Y(:,1), Y(:,2),'b-',...
Z(:,1),Z(:,2),'b:');
text(X(:,1), X(:,2),('abcdefghij')')
text(Y(:,1), Y(:,2),('abcdefghij')')
text(Z(:,1), Z(:,2),('abcdefghij')')
legend('X = Target','Y = Comparison','Z = Transformed','location','SW')
set(gca,'YLim',[0 55],'XLim',[0 65]);

%% Example Hierarchical Clustiuurz

% Compute four clusters of the Fisher iris data using Ward linkage
% and ignoring species information, and see how the cluster
% assignments correspond to the three species.
load fisheriris
Z = linkage(meas,'ward','euclidean');
c = cluster(Z,'maxclust',4);
crosstab(c,species)
dendrogram(Z)