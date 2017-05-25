%% Robohow Dough Rolling Experiments
clear all
clc
d = '/home/nadiafigueroa/dev/MATLAB/michaelchughes-NPBayesHMM/code/demo/data/DoughRolling/';
% d = './';
% Series #
subdir = strcat(d,'Series3/');

files = dir(strcat(subdir,'*.txt'));
data = {};
for ii=1:length(files)
    filename = strcat(subdir,files(ii,1).name);    
    raw_data{ii,1} = textread(filename);
end

data_length = size(raw_data{ii,1},2);

%% KUKA Extract Poses/Forces

EE_CART_Pn = {};
EE_CART_On = {};
EE_FTn = {};
EE_Hn = {};
Table_Hn = {};
for jj=1:length(raw_data)
    poses = [];
    meas = raw_data{jj};
    
    %EE CART POSITION in Base Frame
    EE_CART_Pn{jj,1} = [meas(:,23:25)'];      
    
    %EE CART ORIENTATION in Base Frame
    EE_CART_On{jj,1} = [meas(:,26:28)'];      
    
    %EE FT
    EE_FTn{jj,1} = [meas(:,29:34)'];      
    
    %EE Full H
    H1i = [meas(:,35:38)'];  
    H1ii = reshape(H1i,1,4,length(H1i));
    H2i = [meas(:,39:42)'];  
    H2ii = reshape(H2i,1,4,length(H2i));
    H12ii = cat(1,H1ii,H2ii);
    H3i = [meas(:,43:46)'];  
    H3ii = reshape(H3i,1,4,length(H3i));
    H13 = cat(1,H12ii,H3ii);
    norm = [0 0 0 1];
    n = repmat(norm',1,length(H13));
    N = reshape(n,1,4,length(H13));
    H = cat(1,H13,N);
    
    EE_Hn{jj,1} = H;

    %Table Full H
    H1i = [meas(:,47:50)'];  
    H1ii = reshape(H1i,1,4,length(H1i));
    H2i = [meas(:,51:54)'];  
    H2ii = reshape(H2i,1,4,length(H2i));
    H12ii = cat(1,H1ii,H2ii);
    H3i = [meas(:,55:58)'];  
    H3ii = reshape(H3i,1,4,length(H3i));
    H13 = cat(1,H12ii,H3ii);
    norm = [0 0 0 1];
    n = repmat(norm',1,length(H13));
    N = reshape(n,1,4,length(H13));
    H = cat(1,H13,N);
    
    Table_Hn{jj,1} = H;
    
end

clear raw_data H H12ii H13 H1i H1ii H2i H2ii H3i H3ii Hpose n N

%% Sub Sample RAW DATA
EE_PnP = {};
EE_OnP = {};
EE_FTnP = {};
EE_HnP = {};

for jj=1:length(EE_Hn)
    P_tmp = EE_CART_Pn{jj,1};
    P_sample = P_tmp(:,1:5:end);
    
    O_tmp = EE_CART_On{jj,1};
    O_sample = O_tmp(:,1:5:end);
    
    FT_tmp = EE_FTn{jj,1};
    FT_sample = FT_tmp(:,1:5:end);
    
    H_tmp = EE_Hn{jj,1};
    H_sample = H_tmp(:,:,1:5:end);
    
    EE_PnP{jj,1} = P_sample;
    EE_OnP{jj,1} = O_sample;
    EE_FTnP{jj,1} = FT_sample;
    EE_HnP{jj,1} = H_sample; 
end


%% Convert to 7D for Segmentation
XnQ = {};
for ii=1:length(EE_Hn)
    H = EE_HnP{ii,1};
    R = H(1:3,1:3,:);
    t = reshape(H(1:3,4,:),3,length(H));
    q = quaternion(R);
    Xq = cat(1,t,q);    
    XnQ{ii,1} = Xq;
end

%% Convert to 6D for Segmentation
Xn = [];
for ii=1:length(EE_Hn)    
    %From pos + or
    X = cat(1,EE_CART_Pn{ii,1},EE_CART_On{ii,1}/3.1416);
    Xn{ii,1} = X;
end

%% Convert to 12/13D for Segmentation
Xn = [];
for ii=1:length(EE_Hn)    
    %From pos + or (angle)
    X = cat(1,EE_CART_Pn{ii,1},EE_CART_On{ii,1}/3.1416);    
    %From pos + or (quaternion)
%     X = XnQ{ii,1}    
    %From ee_ft
    FT = cat(1,EE_FTn{ii,1}(1:3,:)/20,EE_FTn{ii,1}(4:6,:)/10);    
    X_tmp = cat(1,X,FT);  
    Xn{ii,1} = X_tmp;
end


%% Check Rotation
Xn_ch = [];
for jj=1:length(Xn)
    %Quaternion
%     X_test = XnQ{jj,1}; rot_end = 7;
    %Angle
    X_test = Xn{jj,1};rot_end = 6;
    X_new = X_test;
    for i=4:1:rot_end
        X_new(i,:) = checkRotation(X_test(i,:));
    end
    Xn_ch{jj,1} = X_new;
    figure;
    subplot(2,1,1)
    plot(X_test')

    subplot(2,1,2)
    plot(X_new')
end

%% Preprocess Data
for jj=1:length(Xn_ch)
    X_tmp = Xn_ch{jj,1};
%     Xn_ch{jj,1} = preprocessMocapData(X_tmp,5,2);     
    X_sample = X_tmp(:,1:5:end);
    Xn_ch{jj,1} = X_sample;
    
end

%% Smooth data
Xn_sm = [];
for ii=1:length(Xn)
    X_tmp = Xn_ch{ii,1};    
    for jj=1:size(X_tmp,1)
        X_tmp(jj,:) = smooth(X_tmp(jj,:),round(size(X_tmp,2)*0.005),'moving');
    end
    Xn_sm{ii,1} = X_tmp;
end
%% Visualize Trajectories + FT
figure('Color',[1 1 1])
n = 1;
subplot(2,1,1)
plot(Xn_ch{n,1}')
legend('x','y','z','roll','pitch','yaw')
subplot(2,1,2)
plot(EE_FTn{n,1}')
legend('fx','fy','fz','tx','ty','tz')
%% Visualize all trajectories
figure('Color',[1 1 1])
% for j=1:3:length(EE_Hn)
for j = 1:1    
% for j=9:12
    h = EE_Hn{j,1};    
    plot3(reshape(h(1,4,:),1,length(h)),reshape(h(2,4,:),1,length(h)),reshape(h(3,4,:),1,length(h)),'Color',[rand rand rand]);
    hold on 
    grid on
end

%% Visualize position trajectories
figure('Name', 'KUKA Dough Rolling','Color',[1 1 1])
for j=1:3:length(EE_Hn)
% for j=1:3:28
% for j=9:12
    h = EE_Hn{j,1};    
    plot3(reshape(h(1,4,:),1,length(h)),reshape(h(2,4,:),1,length(h)),reshape(h(3,4,:),1,length(h)),'Color',[rand rand rand],'LineWidth', 2);
    hold on 
end
xlabel('x');ylabel('y');zlabel('z');
grid on
axis equal

%% Visualize Rigid Body Trajectories
t= 1;
named_figure('Dough Rolling Trial Raw ', t);
clf;
drawframetraj(EE_Hn{t},0.01,1);

t= 12;
named_figure('Dough Rolling Trial Raw ', t);
clf;
drawframetraj(EE_Hn{t},0.01,1);


%% Preprocess for segmentation (From MoCap data pre-processing experiment)
 
RobotChannelNames = {'pos.x','pos,y','pos.z', 'pos.qw','pos.qi','pos.qj','pos.qz'};
% RobotChannelNames = {'pos.x','pos,y','pos.z', 'pos.roll','pos.pitch','pos.yaw'};


N = length(Xn);
nDim = 12;
% nDim = 9;
% nDim = 6;
% nDim = 3;
% nDim = 4;
Preproc.nObj = N;
Preproc.obsDim = nDim;
Preproc.R = 1;
Preproc.windowSize = nDim;
Preproc.channelNames = RobotChannelNames;

% Create data structure
Robotdata = SeqData();
RobotdataN = SeqData();

% Create data structure for PR2 Code
data_struct = struct;

% Read in every sequence, one at a time
for ii = 1:N
    
    % Pos
%     D = Xn_ch{ii}(1:3,:);
    
    % Pos + Or
%     D = Xn_ch{ii}(1:6,:);
  
    %  F + T
%     D = Xn_ch{ii}(7:end,:);
    
    % Pos + F + T
%     D = cat(1,Xn_ch{ii}(1:3:end,:),Xn_ch{ii}(7:end,:));

    % Pos + Or + F + T
    D = Xn_ch{ii};

    % Enforce zero-mean
    D = D';
    D = bsxfun( @minus, D, mean(D,2) );
    D = D';
   

%     D = preprocessMocapData(D, Preproc.windowSize,2);
    %Scale each component of the observation vector so empirical variance
    %Renormalize so for each feature, the variance of the first diff is 1.0
    %of 1st difference measurements is = 1
%     varFirst =  var(diff(D'));
%     for jj=1:length(varFirst)
%         D(jj,:) = D(jj,:)./sqrt(varFirst(jj));
%     end   
%    
%     Tints = Tn{ii,1};
%     labels = Ztrue{ii,1};
%     for jj=1:length(Tints)
%         if jj==1
%             Truelabels = repmat(labels(jj),[1 Tints(jj)]); 
%         else
%             temp = repmat(labels(jj),[1 Tints(jj)]);
%             Truelabels = [Truelabels temp];
%         end
%     end
%     Truelabels = ones(1,length(D))*ii; 
%     TL{ii,1} = zeros(1,length(D)); 
    % Add the sequence to our data structure
    Robotdata = Robotdata.addSeq( D, num2str(ii));
    data_struct(ii).obs = D;
%     data_struct(ii).true_labels = Truelabels;
end

% -------------------------------------- Preproc autoregressive data
% This step simply builds necessary data structs for efficient AR inference
%   including filling in XprevR data field so that for any time tt,
%       XprevR(:,tt) = [Xdata(:, tt-1); Xdata(:,tt-2); ... Xdata(:,tt-R) ]
%   allowing of course for discontinuities at sequence transitions
RobotdataAR = ARSeqData( Preproc.R, Robotdata);

%% Clean up oversegmentations
% for i=1:length(Segm_results)
for i=3:3
    Segments = Segm_results{i}
    Traj = Xn_ch{i,1};
    SegPoints_beg = [1; Segments(:,2)]
    SegPoints_end = [Segments(:,2); length(Traj)]
    Seg_length = SegPoints_beg-SegPoints_end
    Segm_results_new = Segments;
    for j=1:(length(Seg_length))
        if abs(Seg_length(j)) < 20
           Segm_results_new(j,1) = Segm_results_new(j-1,1); 
        end
    end
end
Segm_results_new
%% Plot Segmented Trajectory
% seq = [1 5 10 15];
seq = [1 7 15];

% fig_row = floor(sqrt(length(seq)));
% fig_col = length(seq)/fig_row;
Origin = eye(4);

for i=1:max(Total_feats)
    c(i,:) = [rand rand rand];
end

table_off = 0.02;
table_width = 0.56;
table_height = 0.75;

% Series 3 Details
Chess_width_left = -0.075;
Chess_height_top = 0.013;
o = [556 14]*2; o_t = normalizeNadia(o',fc,cc,kc,alpha_c);
dough = [378 230]*2; dough_t = normalizeNadia(dough',fc,cc,kc,alpha_c);
dough_d1= [360 190]*2; dough_d2 = [360 190]*2;
dough_d1_t = normalizeNadia(dough_d1',fc,cc,kc,alpha_c);
dough_d2_t = normalizeNadia(dough_d2',fc,cc,kc,alpha_c);
cxy = [cc(1) cc(2)]; c_t = normalizeNadia(cxy',fc,cc,kc,alpha_c);

Tc_ext = [ -73.512161 	 -481.637259 	 1617.853582 ]*0.001;
% omc_ext = [ 2.101820 	 2.180334 	 -0.119805 ]
Rc_ext = [ -0.034981 	 0.999078 	 0.024870; 0.990339 	 0.037995 	 -0.133362; -0.134184 	 0.019965 	 -0.990755 ];
% Chess_Origin_Ext = eye(4);
% Chess_Origin_Ext(1:3,1:3) = Rc_ext;
% Chess_Origin_Ext(1:3,4) = 0.001*Tc_ext';

% Series 2 Details
% Chess_width_left = -0.075
% Chess__height_top = 0.031

display('b loop')
for j=1:length(seq)
    figure('Color',[1 1 1])
%     subplot(fig_row,fig_col,j)
    Table_Origin = Table_Hn{j,1}(:,:,1);
    Table_Edge1 = eye(4); Table_Edge2 = eye(4); Table_Edge3 = eye(4); Table_Edge4 = eye(4);
    Table_Edge1(1:3,4) = [-table_off 0 table_off]'; Table_Edge1 = Table_Origin*Table_Edge1;
    Table_Edge2(1:3,4) = [0 0 -table_width]'; Table_Edge2 = Table_Edge1*Table_Edge2;
    Table_Edge3(1:3,4) = [table_height 0 -table_width]'; Table_Edge3 = Table_Edge1*Table_Edge3;
    Table_Edge4(1:3,4) = [table_height 0 0]'; Table_Edge4 = Table_Edge1*Table_Edge4;
    
    Chess_Origin = eye(4); Chess_Origin(1:3,1:3) = rotx(-pi/2); Chess_Origin(1:3,4) = [Chess_height_top 0 -(table_width+Chess_width_left)]'; Chess_Origin = Table_Edge1*Chess_Origin;
    
    Camera_Origin = eye(4); Camera_Origin(1:3,1:3) = Rc_ext^-1; Camera_Origin(1:3,4) = [0 0 0]'; Camera_Origin = Chess_Origin*Camera_Origin;
    T_Cam_Origin = eye(4); T_Cam_Origin(1:3,4) = [-o_t(1)/1.1 -o_t(2)/1.1 0]'; Camera_Origin(1:3,4) = Camera_Origin(1:3,4) + [-o_t(1) -o_t(2) 0]';
 
    Dough_Center = eye(4); Dough_Center(1:3,1:3) = Rc_ext; Dough_Center(1:3,4) = [dough_d1_t(1)/1.1 dough_d1_t(2)/1.1 0]';
    Dough_Center = Camera_Origin*Dough_Center;
    
    TrajSeg = [];
    TrajSeg = Segm_results{seq(j),1};
    Traj = Xn_ch{seq(j),1};
    SegPoints = [1; TrajSeg(:,2)]
    for i=1:length(TrajSeg)
        plot3(Traj(1,SegPoints(i):SegPoints(i+1)),Traj(2,SegPoints(i):SegPoints(i+1)),Traj(3,SegPoints(i):SegPoints(i+1)),'Color', c(TrajSeg(i,1),:),'LineWidth', 3);
        hold on
        text(Traj(1,SegPoints(i)),Traj(2,SegPoints(i)),Traj(3,SegPoints(i))+0.05,num2str(TrajSeg(i,1)),'FontSize',16, 'Color', c(TrajSeg(i,1),:));
        if TrajSeg(i,1)==1
            drawframe(EE_HnP{seq(j),1}(:,:,SegPoints(i)),0.03)
            drawframe(EE_HnP{seq(j),1}(:,:,SegPoints(i+1)),0.03)
        end
        
        if TrajSeg(i,1)==4
            drawframe(EE_HnP{seq(j),1}(:,:,SegPoints(i)),0.015)
            drawframe(EE_HnP{seq(j),1}(:,:,SegPoints(i+1)),0.015)
        end
        
    end
    scatter3(Traj(1,1),Traj(2,1),Traj(3,1), 100, [0 1 0], 'filled')
    scatter3(Traj(1,end),Traj(2,end),Traj(3,end),100, [1 0 0], 'filled')
    
    drawframe(Table_Origin,0.07)
    drawframe(Table_Edge1,0.03)
    drawframe(Table_Edge2,0.03)
    drawframe(Table_Edge3,0.03)
    drawframe(Table_Edge4,0.03)
    drawframe(Chess_Origin,0.03)
    drawframe(Camera_Origin,0.03)
    drawframe(Dough_Center,0.03)
    fill3([Table_Edge1(1,4) Table_Edge2(1,4) Table_Edge3(1,4) Table_Edge4(1,4)],[Table_Edge1(2,4) Table_Edge2(2,4) Table_Edge3(2,4) Table_Edge4(2,4)],[Table_Edge1(3,4) Table_Edge2(3,4) Table_Edge3(3,4) Table_Edge4(3,4)],[0.5 0.5 0.5])
    
    drawframe(Origin,0.1)
        
    str = strcat('Sequence ',num2str(seq(j)));
    title(str);
    xlabel('x');ylabel('y');zlabel('z');
    axis equal
    grid on
end
 