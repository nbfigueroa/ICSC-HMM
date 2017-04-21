%% Paper Example ellipsoids
% close all
% clear all
figure('Color',[1 1 1]);

angle = pi/2;
%Ellipsoid 1
Cov = ones(3,3) + diag([1 1 1]);
mu = [1 1 2]';
% mu = [0 0 0]';
behavs_theta{1,1} = Cov;
[V1,D1] = eig(Cov)
% CoordRot = rotx(angle/3)*roty(angle/3)*rotz(angle/3);
CoordRot = rotx(-angle);
% CoordRot = eye(3);
V1_rot = CoordRot*V1;
Covr1 = V1_rot*D1*V1_rot'
[V1,D1] = eig(Covr1)
[x,y,z] = created3DgaussianEllipsoid(mu,V1,D1^1/2);
mesh(x,y,z,'EdgeColor','blue','Edgealpha',0.2);
hidden off
hold on;

%Arrow Drawung
W1 = eye(4);
W1(1:3,1:3) = V1;
W1(1:3,4) = mu;
drawframe(W1,0.3) 
hold on

X1 = V1*D1^1/2;
triangle = [];
for i=1:3
draw3dArrowN([0 0 0]', mu + X1(:,i),1.5,'b')
triangle(:,i) = mu + X1(:,i);
hold on
end

fill3(triangle(1,:),triangle(2,:),triangle(3,:),'b')
hold on

%Ellipsoid 2: Scale+Noise
D1m = diag(D1)*1.3 + abs(randn(3,1).*[0.35 0.37 0.3]');
D1m = diag(D1)*0.5;
Covs2 = V1*(diag(D1m))*V1';
mu = [0 0 0]';
behavs_theta{1,2} = Covs2;
[V2,D2] = eig(Covs2)

%Ellipsoid 3: Rotated Coordinates
CoordRot = rotx(angle)*roty(angle*0.5)*rotx(angle*2);
CoordRot = eye(3);
CoordRot = rotx(angle);
V2_rot = CoordRot*V2;
Covs3 = V2_rot*D2*V2_rot'
behavs_theta{1,3} = Covs3;
[V3,D3] = eig(Covs3)
% mu = [-1 -1 -2]';
mu = [1 1 -1]';
% mu = [0 0 0]';
[x,y,z] = created3DgaussianEllipsoid(mu,V3,D3^1/2);
mesh(x,y,z,'EdgeColor','red','Edgealpha',0.2);
hidden off
hold on;

% Arrow Drawing
W3 = eye(4);
W3(1:3,1:3) = V3;
W3(1:3,4) = mu;
drawframe(W3,0.3) 
hold on
X3 = V3*D3^1/2;
triangle3 = [];
for i=1:3
draw3dArrowN([0 0 0]', mu + X3(:,i),1.5,'r')
triangle3(:,i) = mu + X3(:,i);
hold on
end

fill3(triangle3(1,:),triangle3(2,:),triangle3(3,:),'r')
hold on


W = eye(4);
drawframe(W,0.5) 

colormap jet
alpha(0.5)
% xlabel('x');ylabel('y');zlabel('z');
% grid off
% axis off
axis equal

%% Underlying Gaussian Computation
clear
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
V1_rot = CoordRot*V1;
Covr1 = V1_rot*D1*V1_rot';
[V1,D1] = eig(Covr1);
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
CoordRot = rotx(angle)*roty(angle*0.5)*rotx(angle*2);
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

%% Matrix Similarity on Grasps
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
        
        spcm(i,j,1) = var(hom_fact); 
        spcm(i,j,2) = mean(hom_fact); 
        spcm(i,j,3) = dir; 
        
   end
end

figure('Color', [1 1 1]);
spcm_cut = spcm(:,:,1);
cut_thres = 10;
for i=1:size(spcm,1)
    for j=1:size(spcm,2)
        if spcm(i,j)>cut_thres
            spcm_cut(i,j) = cut_thres;
        end
    end
end

imagesc(spcm_cut)
title('SPCM Confusion Matrix')
colormap(copper)
colormap(pink)
colorbar 


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

%% Underlying Gaussian Computation of Example Ellipsoids
clear
figure('Color',[1 1 1]);
Cov = [1 0.5 0.3    
       0.5 2 0
       0.3 0 3];
%%%%%%% Ellipsoids %%%%%%%
% Cov = eye(3);
Cov = ones(3,3) + eye(3)
mu = [0 0 0]';
behavs_theta{1,1} = Cov;
[V1,D1] = eig(Cov);
angles = computeShapeRepresentation(D1,V1);
[x,y,z] = created3DgaussianEllipsoid(mu,V1,D1^1/2);
mesh(x,y,z,'EdgeColor','blue','Edgealpha',0.2);
hidden off
hold on;

% Cov Scaled*
D_2 = diag(D1)*2 + abs(randn(3,1).*[0.5 0.7 0.3]');
Covs2 = V1*(diag(D_2))*V1';
behavs_theta{1,2} = Covs2;
[V2,D2] = eig(Covs2);
[x,y,z] = created3DgaussianEllipsoid(mu+[0 5 0]',V2,D2^1/2);
mesh(x,y,z,'EdgeColor','black','Edgealpha',0.2);
hidden off
hold on;


angle = pi/2;
CoordRot = rotx(angle)*roty(angle*0.5)*rotx(angle*2);
V2_rot = CoordRot*V2;
Covs2_rot = V2_rot*D2*V2_rot';
behavs_theta{1,3} = Covs2_rot;
[V2_rot,D2_rot] = eig(Covs2_rot);
[x,y,z] = created3DgaussianEllipsoid(mu+[0 10 0]',V2_rot,D2_rot^1/2);
mesh(x,y,z,'EdgeColor','red','Edgealpha',0.2);
hidden off
hold on;



% Compute SPCM function Confusion Matrix
spcm = [];
dir = 1;
for i=1:length(behavs_theta)
  for j=1:length(behavs_theta)     
        [Vi, Di] = eig(behavs_theta{i});
                
        [Vj, Dj] = eig(behavs_theta{j});
        
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
                
        mean(hom_fact_ij)
        mean(hom_fact_ji)
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

spcm

%% Homothetic principle
figure('Color',[1 1 1]);
Ti = D1^1/2;
Tj = D3^1/2;
fill3(Ti(1,:),Ti(2,:),Ti(3,:),'b')
hold on
for i=1:3
draw3dArrowN([0 0 0]', Ti(:,i),1.5,'b')
hold on
end
fill3(Tj(1,:),Tj(2,:),Tj(3,:),'r')
hold on
for i=1:3
draw3dArrowN([0 0 0]', Tj(:,i),1.5,'r')
hold on
end

[x,y,z] = created3DgaussianEllipsoid([0 0 0]',eye(3),D3^1/2);
mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
hidden off
hold on;

[x,y,z] = created3DgaussianEllipsoid([0 0 0]',eye(3),D1^1/2);
mesh(x,y,z,'EdgeColor','blue','Edgealpha',0.1);
hidden off
hold on;

colormap jet
alpha(0.1)
% xlabel('x');ylabel('y');zlabel('z');
axis off
axis equal

%% Shape Representation of Ellipsoids
figure('Color',[1 1 1])
[angles1 points1] = computeShapeRepresentation(D1,V1);
fill3(points1(1,:),points1(2,:),points1(3,:),'r')
hold on
[angles3 points3] = computeShapeRepresentation(D3,V3);
fill3(points3(1,:),points3(2,:),points3(3,:),'c')
hold on
W = eye(4);
drawframe(W,0.1) 

grid on
xlabel('x');ylabel('y');zlabel('z');
axis equal

%% plot 3D ellipsoid
% developed from the original demo by Rajiv Singh
% http://www.mathworks.com/matlabcentral/newsreader/view_thread/42966
% 5 Dec, 2002 13:44:34
% Example data (Cov=covariance,mu=mean) is included.
% 
% Cov = [1 0.5 0.3
%        0.5 2 0
%        0.3 0 3];
% mu = [1 2 3]';

% X=[];Y=[];Z=[];
j = 3;
behav = behavs{j};
% figure;
% name = strcat('Behavior ', num2str(j),'_',num2str(j),' Cov ellipsoid');
D_behavs = [];
for i=1:length(behav)
mu = behavs_theta{j,i}.mu;
mu = [0 0 0];
Cov = behavs_theta{j,i}.Sigma;

%Smith stuff
% syms x;
% A = x*eye(size(Cov)) - Cov;
% [T,D_beh,S] = smithFormPoly(A);
% D_behavs{i} = precisionCheck(D_beh);

[V,D] = eig(Cov);
% D: eigenvalue diagonal matrix
% V: eigen vector matrix, each column is an eigenvectorc

%Eigen Stuff
eigstuff.V = V;
eigstuff.D = D;
Eig_behavs{i} = eigstuff;

[x,y,z] = created3DgaussianEllipsoid(mu,V,D);

figure('Color',[1 1 1]);
surfl(x,y,z);
xlabel('x');ylabel('y');zlabel('z');
name = strcat('Behavior ', num2str(behavs{j}(i)),' Cov ellipsoid');
title(name)
axis equal
colormap hot;
end

%% Compare angles
corr_prob = [];
dmax = sqrt(prod(size(angles1)));
for i=1:length(angles)
    for j=1:length(angles)
        diff (i,j) = norm(sort(angles{i})-sort(angles{j}),'fro')/dmax;
        corr_prob(i,j) =  exp(-2*pi/dmax*norm(sort(angles{i})-sort(angles{j}),'fro'));
        % Gaussian proximity probability
%         corr_prob(i,j) = exp(-(1/(2*k))*diff(i,j));
      
    end
end

diff
corr_prob

figure('Color',[1 1 1])
imshow(diff, 'InitialMagnification',10000)  % # you want your cells to be larger than single pixels
title('Frobenius Difference of Eigenpolytope Angles')
colormap(hot) % # to change the default grayscale colormap
colorbar


%% Example ellipsoids
clear all
Cov = [1 0.5 0.3
       0.5 2 0
       0.3 0 3];
%%%%%%% Ellipsoids %%%%%%%
% Cov = eye(3);
Cov = ones(3,3) + eye(3)
mu = [0 0 0]';
behavs_theta{1,1} = Cov;
[V1,D1] = eig(Cov)
% X1 = D^1/2*V1'
% Y1 = computeAngleCosineMatrix(X1);
angles = computeShapeRepresentation(D1,V1);
eig1 = diag(D1)
% V = eye(size(D));
[x,y,z] = created3DgaussianEllipsoid(mu,V1,D1);
% [x,y,z] = created3DgaussianEllipsoid(mu,eye(size(V),D);
figure('Color',[1 1 1]);
% subplot(1,2,1)
surfl(x,y,z);
% colormap hot;
% xlabel('x');ylabel('y');zlabel('z');
% axis equal

% Cov Scaled*
% D_perm = diag(D_eig(randperm(3)))
D_2 = diag(D1)*2 + abs(randn(3,1).*[0.5 0.7 1]');
Covs2 = V1*(diag(D_2))*V1'
behavs_theta{1,2} = Covs2;
[V2,D2] = eig(Covs2)
X2 = D2^1/2*V2'
Y2 = computeAngleCosineMatrix(X2);
eig2 = diag(D2)
% V = eye(size(D));
[x,y,z] = created3DgaussianEllipsoid(mu+[0 5 0]',V2,D2);
% [x,y,z] = created3DgaussianEllipsoid(mu+[0 5 0]',eye(size(D)),D);
% subplot(1,2,2)
hold on;
surfl(x,y,z);


angle = pi/2;
CoordRot = rotx(angle)*roty(angle*0.5)*rotx(angle*2);
V2_rot = CoordRot*V2;
Covs2_rot = V2_rot*D2*V2_rot'
behavs_theta{1,3} = Covs2_rot;
[V2_rot,D2_rot] = eig(Covs2_rot)
X2_rot = D2_rot^1/2*V2_rot'
Y2_rot = computeAngleCosineMatrix(X2_rot);
eig2_rot = diag(D2_rot)
[x,y,z] = created3DgaussianEllipsoid(mu+[0 10 0]',V2_rot,D2_rot);
hold on;
surfl(x,y,z);

colormap hot;
xlabel('x');ylabel('y');zlabel('z');
axis equal

% V1 = eye(size(D));
% V2 = eye(size(D));
% V2_rot = eye(size(D));
[angles1, angles2] = pairEigenShape(eig1, V1, eig2, V2,'r')
angles1n = computeShapeRepresentation(D1,V1)
angles2n = computeShapeRepresentation(D2,V2)
[angles2, angles2_rot] = pairEigenShape(eig2, V2, eig2_rot, V2_rot,'g')
angles2n_rot = computeShapeRepresentation(D2_rot,V2_rot)
% 
% D_2d = (ones(3,1)*2 + abs(randn(3,1))).*[1 1 100]';
% Cov_s2d = V*(diag(D_2d))*V^-1;
% behavs_theta{1,3} = Cov_s2d;
% [V,D] = eig(Cov_s2d)
% [x,y,z] = created3DgaussianEllipsoid(mu+[0 10 0]',eye(size(D)),D);
% % subplot(1,2,2)
% hold on;
% surfl(x,y,z);
% 
% D_2dd = abs(diag(D)*0.25 - randn(3,1)*0.1);
% Cov_s2dd = V*(diag(D_2dd))*V^-1;
% behavs_theta{1,4} = Cov_s2dd;
% [V,D] = eig(Cov_s2dd)
% [x,y,z] = created3DgaussianEllipsoid(mu+[0 15 0]',eye(size(D)),D);
% % subplot(1,2,2)
% hold on;
% surfl(x,y,z);
% 
% D_2d = (ones(3,1)*2 + abs(randn(3,1))).*[5 0.001 5]';
% Cov_s2d = V*(diag(D_2d))*V^-1;
% behavs_theta{1,5} = Cov_s2d;
% [V,D] = eig(Cov_s2d)
% [x,y,z] = created3DgaussianEllipsoid(mu+[0 20 0]',eye(size(D)),D);
% % subplot(1,2,2)
% hold on;
% surfl(x,y,z);
% 
% D_2dd = abs(diag(D)*0.25 - randn(3,1)*0.01);
% Cov_s2dd = V*(diag(D_2dd))*V^-1;
% behavs_theta{1,6} = Cov_s2dd;
% [V,D] = eig(Cov_s2dd)
% [x,y,z] = created3DgaussianEllipsoid(mu+[0 26 0]',eye(size(D)),D);
% % subplot(1,2,2)
% hold on;
% surfl(x,y,z);

%% Planes
A = [1 1 0;0 0 1]; Ashift = [1 1 1];
B = [1 1 1]; C = [1 1 0]
figure('Color',[1 1 1])
plotPlanes(A,'d',Ashift,B,'Normal',C)
%%
pointA = [0,0,0];
pointB = [-10,-20,10];
pointC = [10,20,10];
points=[pointA' pointB' pointC']; % using the data given in the question
figure('Color',[1 1 1])
fill3(points(1,:),points(2,:),points(3,:),'r')
grid on
alpha(0.3)
