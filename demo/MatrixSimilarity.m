%% Plot Weakly Grouped Segments
fig_row = floor(sqrt(size(ExtSeg,1)));
fig_col = size(ExtSeg,1)/fig_row;
figure('Color',[1 1 1])
for i=1:size(ExtSeg,1)
    subplot(fig_row,fig_col,i)
    for j=1:size(ExtSeg,2)
        segm = ExtSeg{i,j};
        if ~isempty(segm)
            XnS = Xn{segm.demo}(:,segm.start:segm.finish);
            XnSegs{i,j} = XnS;
            plot3(XnS(1,:),XnS(2,:),XnS(3,:),'Color',[rand rand rand],'LineWidth', 3);
            hold on
            grid on
        end
    end
    tit = strcat('Behavior ', num2str(Total_feats(i)));
    title(tit)
    xlabel('x');ylabel('y');zlabel('z');
    axis equal
end
%% Matrix Similarity on Toy Dataset
parabloid = [17, 15];
line = [6,1,13,12];
helix = [7,8];
sign = [5,16];
screw = [4,3,9,14,11];

dim = 7;
shrinkage = 0;
behavs = {parabloid;line;helix;sign;screw};
behavs_theta = [];
for i=1:length(behavs)
    behav = behavs{i};
    for j=1:length(behav)
        model.mu = bestGauPsi.theta(behav(j)).mu(1:dim,1);
        Sigma = bestGauPsi.theta(behav(j)).invSigma(1:dim,1:dim)^-1;
        if cond(Sigma)>10^4 && shrinkage
            delta = 0.1;
            c = 0.1;
            Sigma_hat = (1-delta)*Sigma + delta*c*eye(dim);
            Sigma = Sigma_hat;
        end
        model.Sigma = Sigma;
        behavs_theta{i,j} = model;
    end
end

%% Matrix Similarity on Carrot Grating
reach_in = [4,5];
adjust = [2];
grate = [1];
reach_out = [3];

dim = 7;
bebavs = [];
behavs = {reach_in;adjust;grate;reach_out};
behavs_theta = [];
for i=1:length(behavs)
    behav = behavs{i};
    for j=1:length(behav)
        model.mu = bestGauPsi.theta(behav(j)).mu(1:dim,1);
        Sigma = bestGauPsi.theta(behav(j)).invSigma(1:dim,1:dim)^-1;
        model.Sigma = Sigma;
        behavs_theta{i,j} = model;
    end
end


%% Matrix Similarity on KuKa Flipping
reach_in = [4,6,7];
swipe = [2];
flip = [1];
back_out = [3,5];

dim = 13;
bebavs = [];
behavs = {reach_in;swipe;flip;back_out};
behavs_theta = [];
for i=1:length(behavs)
    behav = behavs{i};
    for j=1:length(behav)
        model.mu = bestGauPsi.theta(behav(j)).mu(1:dim,1);
        Sigma = bestGauPsi.theta(behav(j)).invSigma(1:dim,1:dim)^-1;
        model.Sigma = Sigma;
        behavs_theta{i,j} = model;
    end
end


%% Matrix Similarity on Carrot Grating Transformed 1 loose
reach_in = [5,4];
% adjust = [2,7];
grate = [3,10,9];
% grate = [9,1];
reach_out = [8,6,2];

%% Matrix Similarity on Carrot Grating Transformed 1/3/4/5

%Transform 3 (Test1)
reach = [2,4,5];
grate = [1,6,3];

%Transform 4 (Test2)
% reach = [2,4,5,6,7];
% grate = [1,3,8];

%Transform 5 (Test3 = Test2 + Scale)
% reach = [3,5,6,7,8];
% grate = [1,2,4];

% Original Data
% reach = [2,3,4,5];
% grate = [1];

dim = 7;
bebavs = [];
behavs = {reach;grate};
behavs_theta = [];
for i=1:length(behavs)
    behav = behavs{i};
    for j=1:length(behav)
        model.mu = bestGauPsi.theta(behav(j)).mu(1:dim,1);
        Sigma = bestGauPsi.theta(behav(j)).invSigma(1:dim,1:dim)^-1;
        model.Sigma = Sigma;
        behavs_theta{i,j} = model;
    end
end

%% Matrix Similarity on Grasps
behavs = [];
behavs_theta = [];
load 6D-Grasps.mat
dim = 6; 
for i=1:size(behavs_theta,1)
    behavs{i,1} = [1:size(behavs_theta,2)] + (i-1)*size(behavs_theta,2);
end

%% plot 3D ellipsoid + shape representations
angles=[];
mu_i = 0.5;
plotStuff = 0;
for j =1:length(behavs)

behav = behavs{j};

if plotStuff==1
    figure('Color',[1 1 1]);
    mu = [0 0 0]';
end

Eig_behavs = [];
for i=1:length(behav)
Cov = behavs_theta{j,i}.Sigma(1:dim,1:dim);
[V,D] = eig(Cov);

%Eigen Stuff
eigstuff.V = V;
eigstuff.D = D;
Eig_behavs{i} = eigstuff;

    if plotStuff==1
        [x,y,z] = created3DgaussianEllipsoid(mu,V,D);
        subplot(2,1,1)
        surfl(x,y,z);
        hold on
        mu = mu + [0 mu_i 0]';
    end
end

if plotStuff==1
    xlabel('x');ylabel('y');zlabel('z');
    name = strcat('Behaviors ', num2str(j),' Cov ellipsoid');
    title(name)
    axis equal
    colormap hot;
end

for i=1:length(Eig_behavs)
%Eigen Stuff
V = Eig_behavs{i}.V;
D = Eig_behavs{i}.D;
[angles_i points] = computeShapeRepresentation(D,V);
color = [rand rand rand];
angles{j,i}=angles_i;
    if plotStuff==1
        subplot(2,1,2)
        fill3(points(1,:),points(2,:),points(3,:),color)
        hold on
    end
end

if dim==3
    xlabel('x');ylabel('y');zlabel('z');
    name = strcat('Shape representations ', num2str(j),' of Cov');
    title(name)
    axis equal
end
end

%% Prepare Data for Full Comparison
Sigmas =[];
Eig_behavs = [];
% Eig_angles = [];
Ids_behavs = [];
k=1;
for i=1:size(behavs_theta,1)
    for j=1:size(behavs_theta,2)
        if ~isempty(behavs_theta{i,j})
            %Real Sigmas
            Sigmas{k}= behavs_theta{i,j}.Sigma(1:dim,1:dim);
       
            %Angles
%             Eig_angles{k} = angles{i,j};
            
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

%% Compare angles
corr_prob = [];
n = size(angles{1,1},1);
m = size(angles{1,1},2);
%Max difference is all 1s
% dmax = sqrt(n*m);

%Max difference according to internal Angles
% SumIntAngles = (n-2)*pi;
% EqualAngle = SumIntAngles/n; 
% dmax = sqrt((cos(EqualAngle)^2)*n*m);
   
%Max difference according to Triangular Face Internal Angles
InnerTriangles = n*m/3;
dmax = sqrt(sum(([0;0;1]-[0.5;0.5;0.5]).^2*InnerTriangles));

% From Max Triangle Difference
max_fro = 1.6583;
mean_fro = 0.5494;
% max_fro2 = ((max_fro+mean_fro)/2)^2;
max_fro2 = (max_fro)^2;
dmax = sqrt(max_fro2*InnerTriangles);

diff_fro = [];
cos_sim = [];
eig_std = [];
eig_ave = [];
c = pi;
clear norm;

% From Max Triangle Difference
max_cos = 0.9415;
X = [];Y = [];

%% Compute SPCM
for i=1:length(Eig_behavs)
    for j=1:length(Eig_behavs)        
%         X = cos(sort(Eig_angles{i}));
%         Y = cos(sort(Eig_angles{j}));
        %Frobenius Norm
%         diff_fro(i,j) = norm(X-Y,'fro')/dmax;
        
        %Cosine Similarity
%         xi = sort(Eig_angles{i}(:));
%         xj = sort(Eig_angles{j}(:));           
%         cosij = (xi'*xj)/(norm(xi)*norm(xj)); 
%         cos_sim(i,j) = (cosij-max_cos)/(1-max_cos);         
        
        %EigenValues Ratio Std
        %Pure Eigenvalues
%         eig_i = diag(Eig_behavs{i}.D^1/2);
%         eig_j = diag(Eig_behavs{j}.D^1/2);
        
        %Norms of Eigenvalue Scaled EigenVectors
        Xi = Eig_behavs{i}.V*Eig_behavs{i}.D^1/2;
        Xj = Eig_behavs{j}.V*Eig_behavs{j}.D^1/2;
        
        
        for k=1:length(Eig_behavs{i}.D)
            eig_i(k,1) = norm(Xi(:,k));
            eig_j(k,1) = norm(Xj(:,k));
        end
        
        %Homothetic factors
        hom_fact_ij = eig_i./eig_j;
        hom_fact_ji = eig_j./eig_i;
        
%         cos_sim(i,j) = (eig_i'*eig_j)/(norm(eig_i)*norm(eig_j));
        eig_std(i,j) =max(std(hom_fact_ij),std(hom_fact_ji)); 
%         eig_ave(i,j) =max(mean(hom_fact_ij),mean(hom_fact_ji)); 
%         eig_range(i,j) =max(range(hom_fact_ij),range(hom_fact_ji)); 
        % Gaussian proximity probability        
%         corr_prob(i,j) =  exp(-c*(diff_fro(i,j)));             
    end
end

% diff_fro;
%% corr_prob;

figure('Color',[1 1 1])
% subplot(1,3,1)
% diff = diff_fro*dmax;
% diff_norm = (diff - min(min(diff)))/(max(max(diff)) - min(min((diff))));
% % imshow(diff_fro, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
% imagesc(diff_fro)
% title('Frobenius Norm of Eigenspace Angles')
% colormap(jet) % # to change the default grayscale colormap
% colorbar
% 
% subplot(1,3,2)
% % imshow(cos_sim, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
% imagesc(cos_sim)
% title('Cosine Similarity of EigenSpace Angles')
% colormap(jet) % # to change the default grayscale colormap
% colorbar
% 
% subplot(1,3,3)
eig_norm = (eig_std - min(min(eig_std)))/(max(max(eig_std)) - min(min((eig_std))));
eig_std2 = eig_std; 
eig_std_cut = eig_std;
cut_thres = 10;
for i=1:size(eig_std,1)
    for j=1:size(eig_std,2)
        if eig_std(i,j)>cut_thres
            eig_std_cut(i,j) = cut_thres;
        end
    end
end

imagesc(eig_std_cut)
% imagesc(eig_std.^2)
% imagesc(eig_std_cut)
% imshow(eig_norm, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
% title('Max EigenValue Ratio Standard Deviation')
title('SPCM Confusion Matrix')
%  colormap(summer)
%  colormap(jet);
% colormap(autumn)
%   colormap(hot)
colormap(copper)
colormap(pink)
colorbar 

% subplot(1,4,4)
% inf_norm = (diff_inf - min(min(diff_inf)))/(max(max(diff_inf)) - min(min((diff_inf))));
% imshow(eig_norm, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
% title('Max EigenValue Ratio Standard Deviation')
% colorbar 

%% Final Similarity
figure('Color',[1 1 1])
alpha = 0.4;
beta = 0.6;
Similarity = alpha*(1-diff_fro) + beta*(cos_sim);
imshow(Similarity, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
title('Final Similarity Measure')
colormap(jet) % # to change the default grayscale colormap
colorbar

%% Group Similar Behaviors

%Toy Dataset
% groups = MotionGrouping(diff_fro,0.09);
% Carrot Transform 3
% groups = MotionGrouping(1-cos_sim,0.02);
% Carrot Transform 3
% groups = MotionGrouping(diff_fro,0.09);
% groups = MotionGrouping(eig_std./eig_ave,0.5);
groups = MotionGrouping(eig_std.^2,1);
% groups = MotionGrouping(Div_lerm,max(max(Div_lerm))*0.05);%toy=0.2,grate=0.4
% groups = MotionGrouping(Div_kldm,max(max(Div_kldm))*0.1);%toy=0.001,grate=0.15
%groups = MotionGrouping(Div_jb,max(max(Div_jb))*0.1);%toy=0.25,grate=0.45,grasp=

%% Comparisons
ids = [1 2 3 5 7 9 14 15]
toy_ids = ids
groups = MotionGrouping(eig_std(toy_ids,toy_ids).^2,1);
groups = MotionGrouping(Div_kldm(toy_ids,toy_ids),max(max(Div_kldm(toy_ids,toy_ids)))*0.1);
%% Plot Grouped Motion Behaviors
clear grouped_behavs
fig_row = floor(sqrt(size(groups,1)));
fig_col = size(groups,1)/fig_row;
figure('Color',[1 1 1])
for ii=1:length(groups)
    g = groups{ii};
    gs = [];
    subplot(fig_row,fig_col,ii)
    for jj=1:length(g)
        behav.g_id = g(jj);
        behav.bt_id = Ids_behavs{g(jj)};
        behav.seg_id = behavs{behav.bt_id(1)}(behav.bt_id(2));
        behav.mu = behavs_theta{behav.bt_id(1),behav.bt_id(2)}.mu;
        behav.Sigma = behavs_theta{behav.bt_id(1),behav.bt_id(2)}.Sigma;
        behav.V = Eig_behavs{g(jj)}.V;
        behav.D = Eig_behavs{g(jj)}.D;
        behav.EigAngles = Eig_angles{g(jj)};
        gs = [gs behav.seg_id];
        XnS = XnSegs{find(Total_feats==behav.seg_id),1};
        behav.Xn = XnS;
        [V3 D3] = eig(behav.Sigma(1:3,1:3));
        behav.V3 = V3;
        behav.D3 = D3;
        behav.t3 = behav.mu(1:3);
        grouped_behavs{ii,jj} = behav;
	    plot3(XnS(1,:),XnS(2,:),XnS(3,:),'Color',[rand rand rand],'LineWidth',2);
%         HnS = fromXtoH(XnS,50);
%         drawframetraj(HnS,1,1);
        hold on
        grid on
        W = eye(4);
        W(1:3,4) = behav.t3;
        W(1:3,1:3) = behav.V3;
        drawframe(W,0.1) 
%         [x,y,z] = created3DgaussianEllipsoid(behav.t3,behav.V3,behav.D3);
%         surfl(x,y,z);
%         hold on   
    end
    tit = strcat('Group ', num2str(ii),' --> Behavs: ',num2str(gs));
    title(tit)
    xlabel('x');ylabel('y');zlabel('z');
    axis equal
end

%% Plot Grasp Ellipsoids
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
        behav.EigAngles = Eig_angles{g(jj)};
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

%% Comparison of Divergence Measure for Covariance Matrices
Div_riem = zeros(length(Sigmas));
Div_lerm = zeros(length(Sigmas));
Div_kldm= zeros(length(Sigmas));
Div_jb = zeros(length(Sigmas));
Eigen = 0;
Eig_stats = zeros(length(Sigmas),length(Sigmas),3);
for i=1:length(Sigmas)
    for j=1:length(Sigmas)
        X = Sigmas{i};
        Y = Sigmas{j};
        % Affine Invariant Riemannian Metric
        Div_riem(i,j) = (norm(logm(X^(-1/2)*Y*X^(-1/2)),'fro')^2);
        % Log-Euclidean Riemannian Metric
        Div_lerm(i,j) = (norm(logm(X) - logm(Y),'fro')^2);
        % Kullback-Liebler Divergence Metric (KDLM)
        Div_kldm(i,j) = 1/2*trace(X^-1*Y + Y^-1*X - 2*eye(size(X)));
        % Jensen-Bregman LogDet Divergence
        Div_jb(i,j) = (logm(det((X+Y)/2)) - 1/2*logm(det(X*Y)));
        
        if Eigen
            % Eigen Statistics
            Vj = Eig_behavs{i}.V;
            Dj = Eig_behavs{i}.D;
            S2=0; S3=0;
            Vk = Eig_behavs{j}.V;
            Dk = Eig_behavs{j}.D;
            V11 = zeros(length(Vj),1);
            V12 = zeros(length(Vk),1);
            V21 = zeros(length(Vk),1);
            V22 = zeros(length(Vk),1);
            for ii=1:length(Vk)
                vi11 = 0;
                vi12 = 0;
                vi21 = 0;
                vi22 = 0;
                for jj=1:length(Vk)
                    vi11 = vi11 + Vj(:,ii)'*Vj(:,jj)*sqrt(Dj(jj,jj));
                    vi12 = vi12 + Vj(:,ii)'*Vk(:,jj)*sqrt(Dk(jj,jj));
                    vi21 = vi21 + Vk(:,ii)'*Vj(:,jj)*sqrt(Dj(jj,jj));
                    vi22 = vi22 + Vk(:,ii)'*Vk(:,jj)*sqrt(Dk(jj,jj));
                end
                V11(ii) = vi11;
                V12(ii) = vi12;
                V21(ii) = vi21;
                V22(ii) = vi22;
            end
            for ss=1:length(Vk)
                S2 = S2 + ((V11(ss)+V22(ss))-(V12(ss)+V21(ss)))^2;
                S3 = S3 + ((V11(ss)+V12(ss))-(V21(ss)+V22(ss)))^2;
            end
            Eig_stats(i,j,1) = (S2 + S3);
            Eig_stats(i,j,2) = S2;
            Eig_stats(i,j,3) = S3;
        end
    end
end

figure('Color',[1 1 1])
tot = length(Sigmas);
subplot(2,2,1)
Dnorm_riem = (Div_riem(1:tot,1:tot) - min(min(Div_riem(1:tot,1:tot))))/(max(max(Div_riem(1:tot,1:tot))) - min(min((Div_riem(1:tot,1:tot)))));
% imshow(Dnorm_riem, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
imagesc(Div_riem)
title('Behavior Covariance Riemannian Divergence Metric')
% colormap(copper) % # to change the default grayscale colormap 
colorbar
subplot(2,2,2)
Dnorm_lerm = (Div_lerm(1:tot,1:tot) - min(min(Div_lerm(1:tot,1:tot))))/(max(max(Div_lerm(1:tot,1:tot))) - min(min((Div_lerm(1:tot,1:tot)))));
% imshow(Dnorm_lerm, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
imagesc(Div_lerm)
title('Behavior Covariance Log-Euclidean Riemannian Divergence Metric')
% colormap(copper) % # to change the default grayscale colormap
colorbar
subplot(2,2,3)
Dnorm_kldm = (Div_kldm(1:tot,1:tot) - min(min(Div_kldm(1:tot,1:tot))))/(max(max(Div_kldm(1:tot,1:tot))) - min(min((Div_kldm(1:tot,1:tot)))));
% imshow(Dnorm_kldm, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
imagesc(Div_kldm)
title('Behavior Covariance Kullback-Leibler Divergence Metric')
% colormap(copper) % # to change the default grayscale colormap
colorbar
subplot(2,2,4)
Dnorm_jb = (Div_jb(1:tot,1:tot) - min(min(Div_jb(1:tot,1:tot))))/(max(max(Div_jb(1:tot,1:tot))) - min(min((Div_jb(1:tot,1:tot)))));
% imshow(Dnorm_jb, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
imagesc(Div_jb)
title('Behavior Covariance Jensen-Bregman Divergence Metric')

colormap(jet) % # to change the default grayscale colormap
colorbar

%% AMI ARI COMPARISONS for alpha
alpha =    [0:0.1:1];
alphac1_ARI= [0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
alphac1_AMI= [0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

alphac2_ARI= [0, 0.5714285714285715, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
alphac2_AMI= [0, 0.40000000000000024, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

alphac3_ARI= [0, 0.6153846153846153, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
alphac3_AMI= [0, 0.44444444444444386, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944]

alphac4_ARI= [0, 0.6341463414634146, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
alphac4_AMI= [0, 0.46428571428571386, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944, 0.99999999999999944]

alphac5_ARI= [0, 0.47169811320754723, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
alphac5_AMI= [0, 0.30864197530864285, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

figure('Color',[1 1 1])
plot(alpha,alphac1_ARI,'bo-',alpha,alphac2_ARI,'gx-',alpha,alphac3_ARI,'rx-',alpha,alphac4_ARI,'c+-',alpha,alphac5_ARI,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\alpha$','Interpreter','LaTex','FontSize',30)
ylabel('Adjusted Rand Index(ARI)','FontSize',30)

figure('Color',[1 1 1])
plot(alpha,alphac1_AMI,'bo-',alpha,alphac2_AMI,'gx-',alpha,alphac3_AMI,'rx-',alpha,alphac4_AMI,'c+-',alpha,alphac5_AMI,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\alpha$','Interpreter','LaTex','FontSize',30)
ylabel('Adjusted Mutual Information(AMI)','FontSize',30)

%% AMI ARI COMPARISONS for sigma jb
sigma =    [0:0.1:1];
sigmac1_ARI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
sigmac1_AMI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

sigmac2_ARI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
sigmac2_AMI= [0, 4.805139755722374e-16, 4.805139755722374e-16, 4.805139755722374e-16, 4.805139755722374e-16, 4.805139755722374e-16, 2.3256314468733853e-16, 2.3256314468733853e-16, 2.3256314468733853e-16, 0.0, 0.0]

sigmac3_ARI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21052631578947364, 0.28571428571428564, 0.28571428571428564, -1.1102230246251568e-16, -1.1102230246251568e-16]
sigmac3_AMI= [0, -1.2012849389305949e-15, -1.2012849389305949e-15, -1.2012849389305949e-15, -1.2012849389305949e-15, -1.2012849389305949e-15, 0.22504228319830813, 0.28571428571428525, 0.28571428571428525, 0.0, 0.0]

sigmac4_ARI= [0, -0.0975609756097561, -0.0975609756097561, -0.0975609756097561, -0.0975609756097561, 0.4230769230769231, 0.18918918918918917, -0.071428571428571411, 0.0, 0.0, 0.0]
sigmac4_AMI= [0, -0.071428571428571203, -0.071428571428571203, -0.071428571428571203, -0.071428571428571203, 0.42307692307692307, 0.16408176832885166, -0.07454641833876674, -8.349667571982862e-17, -8.349667571982862e-17, -8.349667571982862e-17]

sigmac5_ARI= [0, -0.056603773584905634, -0.056603773584905634, -0.056603773584905634, 0.14110429447852765, 0.16513761467889918, 0.16513761467889918, 0.16513761467889918, 1.0658141036401501e-16, 1.0658141036401501e-16, 1.0658141036401501e-16]
sigmac5_AMI= [0, -0.037037037037035689, -0.037037037037035689, -0.037037037037035689, 0.14599933265768855, 0.18261426003782535, 0.18261426003782535, 0.18261426003782535, 0.0, 0.0, 0.0]

figure('Color',[1 1 1])
plot(sigma,sigmac1_ARI,'bo-',sigma,sigmac2_ARI,'gx-',sigma,sigmac3_ARI,'rx-',sigma,sigmac4_ARI,'c+-',sigma,sigmac5_ARI,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\sigma_{max}$','Interpreter','LaTex','FontSize',30)
ylabel('Adjusted Rand Index(ARI)','FontSize',30)

figure('Color',[1 1 1])
plot(sigma,sigmac1_AMI,'bo-',sigma,sigmac2_AMI,'gx-',sigma,sigmac3_AMI,'rx-',sigma,sigmac4_AMI,'c+-',sigma,sigmac5_AMI,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\sigma_{max}$','Interpreter','LaTex','FontSize',30)
ylabel('Adjusted Mutual Information(AMI)','FontSize',30)
%% AMI ARI COMPARISONS for sigma lerm
sigma =    [0:0.1:1];

sigmac1_ARI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
sigmac1_AMI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

sigmac2_ARI= [0, 0.0, 0.0, 0.0, 0.5714285714285715, 0.5714285714285715, 0.0, 0.0, 0.0, 0.0, 0.0]
sigmac2_AMI= [0, 4.805139755722374e-16, 4.805139755722374e-16, 4.805139755722374e-16, 0.40000000000000024, 0.40000000000000024, 2.3256314468733853e-16, 2.3256314468733853e-16, 2.3256314468733853e-16, 0.0, 0.0]

sigmac3_ARI= [0, 0.0, 0.0, 0.0, 0.6153846153846153, 0.21052631578947364, -0.071428571428571494, -0.071428571428571494, -1.1102230246251568e-16, -1.1102230246251568e-16, -1.1102230246251568e-16]
sigmac3_AMI= [0, -1.2012849389305949e-15, -1.2012849389305949e-15, -1.2012849389305949e-15, 0.44444444444444375, 0.22504228319830813, -0.071428571428572063, -0.071428571428572063, 0.0, 0.0, 0.0]

sigmac4_ARI= [0, -0.0975609756097561, -0.0975609756097561, -0.0975609756097561, 0.062499999999999986, -0.071428571428571411, -0.071428571428571411, -0.071428571428571411, 0.0, 0.0, 0.0]
sigmac4_AMI= [0, -0.071428571428571203, -0.071428571428571203, -0.071428571428571203, 0.062499999999999674, -0.07454641833876674, -0.07454641833876674, -0.07454641833876674, -8.349667571982862e-17, -8.349667571982862e-17, -8.349667571982862e-17]

sigmac5_ARI= [0, -0.056603773584905634, 0.34375000000000006, 0.29411764705882365, 0.14110429447852765, 0.14110429447852765, 0.16513761467889918, 1.0658141036401501e-16, 1.0658141036401501e-16, 1.0658141036401501e-16, 1.0658141036401501e-16]
sigmac5_AMI= [0, -0.037037037037035689, 0.28205128205128299, 0.29411764705882376, 0.14599933265768855, 0.14599933265768855, 0.18261426003782535, 0.0, 0.0, 0.0, 0.0]

figure('Color',[1 1 1])
plot(sigma,sigmac1_ARI,'bo-',sigma,sigmac2_ARI,'gx-',sigma,sigmac3_ARI,'rx-',sigma,sigmac4_ARI,'c+-',sigma,sigmac5_ARI,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\sigma_{max}$','Interpreter','LaTex','FontSize',30)
ylabel('Adjusted Rand Index(ARI)','FontSize',30)

figure('Color',[1 1 1])
plot(sigma,sigmac1_AMI,'bo-',sigma,sigmac2_AMI,'gx-',sigma,sigmac3_AMI,'rx-',sigma,sigmac4_AMI,'c+-',sigma,sigmac5_AMI,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\sigma_{max}$','Interpreter','LaTex','FontSize',30)
ylabel('Adjusted Mutual Information(AMI)','FontSize',30)
%% AMI ARI COMPARISONS for sigma kldm
sigma =    [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075, 0.1:0.1:0.7, 1];

sigmac1_ARI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
sigmac1_AMI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

sigmac2_ARI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5714285714285715, 0.5714285714285715, 0.5714285714285715, 0.5714285714285715, 0.5714285714285715, 0.5714285714285715, 0.5714285714285715, 0.0, 0.0]
sigmac2_AMI= [0, 4.805139755722374e-16, 4.805139755722374e-16, 4.805139755722374e-16, 4.805139755722374e-16, 4.805139755722374e-16, 4.805139755722374e-16, 4.805139755722374e-16, 0.40000000000000024, 0.40000000000000024, 0.40000000000000024, 0.40000000000000024, 0.40000000000000024, 0.40000000000000024, 0.40000000000000024, 0.0, 0.0]

sigmac3_ARI= [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6153846153846153, 0.6153846153846153, 0.21052631578947364, 0.21052631578947364, 0.21052631578947364, 0.21052631578947364, 0.21052631578947364, -1.1102230246251568e-16, -1.1102230246251568e-16]
sigmac3_AMI= [0, -1.2012849389305949e-15, -1.2012849389305949e-15, -1.2012849389305949e-15, -1.2012849389305949e-15, -1.2012849389305949e-15, -1.2012849389305949e-15, -1.2012849389305949e-15, 0.44444444444444375, 0.44444444444444375, 0.22504228319830835, 0.22504228319830835, 0.22504228319830835, 0.22504228319830835, 0.22504228319830835, 0.0, 0.0]

sigmac4_ARI= [0, -0.0975609756097561, -0.0975609756097561, -0.0975609756097561, -0.0975609756097561, -0.0975609756097561, 0.4230769230769231, 0.4230769230769231, 0.062499999999999986, 0.062499999999999986, 0.062499999999999986, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
sigmac4_AMI= [0, -0.071428571428571203, -0.071428571428571203, -0.071428571428571203, -0.071428571428571203, -0.071428571428571203, 0.42307692307692307, 0.42307692307692307, 0.062499999999999674, 0.062499999999999674, 0.062499999999999674, -8.349667571982862e-17, -8.349667571982862e-17, -8.349667571982862e-17, -8.349667571982862e-17, -8.349667571982862e-17, -8.349667571982862e-17]

sigmac5_ARI= [0, -0.056603773584905634, 0.6266666666666667, 0.29411764705882365, 0.29411764705882365, 0.29411764705882365, 0.16513761467889918, 0.16513761467889918, 0.16513761467889918, 1.0658141036401501e-16, 1.0658141036401501e-16, 1.0658141036401501e-16, 1.0658141036401501e-16, 1.0658141036401501e-16, 1.0658141036401501e-16, 1.0658141036401501e-16, 1.0658141036401501e-16]
sigmac5_AMI= [0, -0.037037037037035689, 0.62666666666666682, 0.29411764705882376, 0.29411764705882376, 0.29411764705882376, 0.18261426003782535, 0.18261426003782535, 0.18261426003782535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

figure('Color',[1 1 1])
plot(sigma,sigmac1_ARI,'bo-',sigma,sigmac2_ARI,'gx-',sigma,sigmac3_ARI,'rx-',sigma,sigmac4_ARI,'c+-',sigma,sigmac5_ARI,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\sigma_{max}$','Interpreter','LaTex','FontSize',30)
ylabel('Adjusted Rand Index(ARI)','FontSize',30)

figure('Color',[1 1 1])
plot(sigma,sigmac1_AMI,'bo-',sigma,sigmac2_AMI,'gx-',sigma,sigmac3_AMI,'rx-',sigma,sigmac4_AMI,'c+-',sigma,sigmac5_AMI,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\sigma_{max}$','Interpreter','LaTex','FontSize',30)
ylabel('Adjusted Mutual Information(AMI)','FontSize',30)
%% Comparisons Homogeneity
alpha =    [0:0.1:1];
alpha_c1 = [0 1 1 1 1 1 1 1 1 1 1];
alpha_c2 = [0 1 1 1 1 1 1 1 1 1 1];
alpha_c3 = [0 1 1 1 1 1 1 1 1 1 1];
alpha_c4 = [0 1 1 1 1 1 1 1 1 1 1];
alpha_c5 = [0 1 1 1 1 1 1 1 1 1 1];

figure('Color',[1 1 1])
plot(alpha,alpha_c1,'bo-',alpha,alpha_c2,'gx-',alpha,alpha_c3,'rx-',alpha,alpha_c4,'c+-',alpha,alpha_c5,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\alpha$','Interpreter','LaTex','FontSize',30)
ylabel('Homogeneity','FontSize',30)

sigma =    [0:0.1:1];
sigma_c1 = [0 1 1 1 1 1 1 1 1 1 1];
sigma_c2 = [0 1 1 1 1 1 0.3112781244591327 0.3112781244591327  0.3437110184854507 0 0];
sigma_c3 = [0 1 1 1 1 1 0.6379740263133316 0.47435098761403194 0.47435098761403194 0 0];
sigma_c4 = [0 0.8262346571285599 0.8262346571285599  0.8262346571285599  0.8262346571285599  0.8262346571285599 0.58688267143572 0.1650887640441101 0 0 0];
sigma_c5 = [0 0.8888888888888888 0.8888888888888888 0.8888888888888888 0.4661310847535105 0.3605680553151701 0.3605680553151701 0.3605680553151701 0 0 0];
figure('Color',[1 1 1])
plot(sigma,sigma_c1,'bo-',sigma,sigma_c2,'gx-',sigma,sigma_c3,'rx-',sigma,sigma_c4,'c+-',sigma,sigma_c5,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\sigma_{max}$','Interpreter','LaTex','FontSize',30)
ylabel('Homogeneity','FontSize',30)

%% Comparisons Completeness
alpha =    [0:0.1:1];
alpha_c1 = [0 0 1 1 1 1 1 1 1 1 1];
alpha_c2 = [0 0.6666666666666666 1 1 1 1 1 1 1 1 1];
alpha_c3 = [0 0.7918756684685216 1 1 1 1 1 1 1 1 1];
alpha_c4 = [0 0.8519590445170674 1 1 1 1 1 1 1 1 1];
alpha_c5 = [0 0.8181818181818181 1 1 1 1 1 1 1 1 1];

figure('Color',[1 1 1])
plot(alpha,alpha_c1,'bo-',alpha,alpha_c2,'gx-',alpha,alpha_c3,'rx-',alpha,alpha_c4,'c+-',alpha,alpha_c5,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\alpha$','Interpreter','LaTex','FontSize',30)
ylabel('Completeness','FontSize',30)

sigma =    [0:0.1:1];
sigma_c1 = [0 0 0 0 0 0 0 0 0 1.0 1.0];
sigma_c2 = [0 0.5 0.5 0.5 0.5 0.5 0.3836885465963443 0.3836885465963443 0.3836885465963443 1.0 1.0];
sigma_c3 = [0 0.6554587535412857 0.6554587535412857 0.6554587535412857 0.6554587535412857 0.6554587535412857 0.708231644803283 1.0 1.0 1 1];
sigma_c4 = [0 0.7039180890341348 0.7039180890341348 0.7039180890341348 0.7039180890341348 0.82623465712856 0.7715561736794712 0.48719717623270187 1 1 1];
sigma_c5 = [0 0.7272727272727273 0.7272727272727273 0.7272727272727273 0.8075138790838333 1 1 1 1 1 1];
figure('Color',[1 1 1])
plot(sigma,sigma_c1,'bo-',sigma,sigma_c2,'gx-',sigma,sigma_c3,'rx-',sigma,sigma_c4,'c+-',sigma,sigma_c5,'m*-','LineWidth',2,'MarkerSize',10)
set(gca,'FontSize',15)
legend('1-class','2-class','3-class','4-class','5-class')
set(gca,'FontSize',15)
xlabel('$\sigma_{max}$','Interpreter','LaTex','FontSize',30)
ylabel('Completeness','FontSize',30)
%% sfd
figure('Color',[1 1 1])
subplot(1,4,1)
imagesc(Div_lerm)
title('Log-Euclidean Riemannian Metric (LERM)')
colorbar

subplot(1,4,2)
imagesc(Div_kldm)
title('Kullback-Leibler Divergence Metric (KLDM)')
colorbar

subplot(1,4,3)
imagesc(Div_jb)
title('Jensen-Bregman LogDet Divergence (JBLD)')
colorbar

subplot(1,4,4)
eig_new = eig_std.^2; 
eig_new(eig_new > 1000) = 1000;
imagesc(exp(-eig_std.^2))
title('Spectral Polytope Covariance Similarity (SPCM)')
colorbar
colormap(jet) % # to change the default grayscale colormap


%% Eigen plots
figure('Color',[1 1 1])
subplot(1,3,1)
Eig_S1 = (Eig_stats(1:tot,1:tot,1) - min(min(Eig_stats(1:tot,1:tot,1))))/(max(max(Eig_stats(1:tot,1:tot,1))) - min(min((Eig_stats(1:tot,1:tot,1)))));
imshow(Eig_S1, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
title('General Eigen Statistic (S1=S2+S3)')
% colormap(copper) % # to change the default grayscale colormap 
colorbar

subplot(1,3,2)
Eig_S2 = (Eig_stats(1:tot,1:tot,2) - min(min(Eig_stats(1:tot,1:tot,2))))/(max(max(Eig_stats(1:tot,1:tot,2))) - min(min((Eig_stats(1:tot,1:tot,2)))));
imshow(Eig_S2, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
title('Orientation Eigen Statistic (S2)')
% colormap(copper) % # to change the default grayscale colormap
colorbar

subplot(1,3,3)
Eig_S3 = (Eig_stats(1:tot,1:tot,3) - min(min(Eig_stats(1:tot,1:tot,3))))/(max(max(Eig_stats(1:tot,1:tot,3))) - min(min((Eig_stats(1:tot,1:tot,3)))));
imshow(Eig_S3, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
title('Shape Eigen Statistic (S3)')

colormap(hot) % # to change the default grayscale colormap
colorbar
%% now plot the rotated ellipses
% sc = surf(x,y,z); shading interp; colormap copper
figure;
subplot(1,2,1)
surfl(X(:,:,1), Y(:,:,1), Z(:,:,1)); 
subplot(1,2,2)
surfl(X(:,:,2), Y(:,:,2), Z(:,:,2)); 
colormap hot;
title('parabloid ellipsoids represented by mu and Cov')
axis equal
alpha(0.7)


%% models = mvn_new(squeeze(behavs_theta{1,1}.invSigma^-1),squeeze(behavs_theta{1,1}.mu));
k=1;
for i=1:size(behavs_theta,1)
    for j=1:size(behavs_theta,2)
        if ~isempty(behavs_theta{i,j})
            models(k) = mvn_new(squeeze(behavs_theta{i,j}.Sigma),squeeze(behavs_theta{i,j}.mu));
            k = k+1;
        end
    end
end
D = mvn_divmat(models, 'skl')*0.00001;
figure('Color',[1 1 1])
imshow(D, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
title('Behavior Covariance symmetrized Kullback-Leibler Divergence Metric')
colormap(copper) % # to change the default grayscale colormap


% [centroids c_assignment] = mvn_kmeans(models, 5, 'skl');
