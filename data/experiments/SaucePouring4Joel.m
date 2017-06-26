%% Robohow Sauce Pouring Experiments
clear all
clc
d = '/home/nadiafigueroa/dev/MATLAB/michaelchughes-NPBayesHMM/code/demo/';
% d = './';
% Series #
subdir = strcat(d,'Pouring_in_two_segments_4Joel/');

files = dir(strcat(subdir,'*.txt'));
data = {};
for ii=1:length(files)
    filename = strcat(subdir,files(ii,1).name);    
    raw_data{ii,1} = textread(filename);
end

data_length = size(raw_data{ii,1},2);

%% KUKA Extract Poses/Forces

time = {};
EE_CART_Pn = {};
EE_CART_On = {};
EE_FTn = {};
EE_Hn = {};
Sensor_FTn = {};
for jj=1:length(raw_data)
    poses = [];
    meas = raw_data{jj};
    
    time{jj,1} = [meas(:,1)'];
    %EE CART POSITION in Base Frame
    EE_CART_Pn{jj,1} = [meas(:,2:4)'];      
    
    %EE CART ORIENTATION in Base Frame
    EE_CART_On{jj,1} = [meas(:,5:7)'];      
    
    %EE FT
    EE_FTn{jj,1} = [meas(:,8:13)'];      
    
    %EE Full H
    H1i = [meas(:,14:17)'];  
    H1ii = reshape(H1i,1,4,length(H1i));
    H2i = [meas(:,18:21)'];  
    H2ii = reshape(H2i,1,4,length(H2i));
    H12ii = cat(1,H1ii,H2ii);
    H3i = [meas(:,22:25)'];  
    H3ii = reshape(H3i,1,4,length(H3i));
    H13 = cat(1,H12ii,H3ii);
    norm = [0 0 0 1];
    n = repmat(norm',1,length(H13));
    N = reshape(n,1,4,length(H13));
    H = cat(1,H13,N);
    
    EE_Hn{jj,1} = H;  
end

clear H H12ii H13 H1i H1ii H2i H2ii H3i H3ii Hpose n N

%% Sub Sample RAW DATA
time_P = {};
EE_PnP = {};
EE_OnP = {};
EE_FTnP = {};
EE_HnP = {};

for jj=1:length(EE_Hn)
    time_tmp = time{jj,1};
    time_sample = time_tmp(:,1:5:end);    
    
    P_tmp = EE_CART_Pn{jj,1};
    P_sample = P_tmp(:,1:5:end);
    
    O_tmp = EE_CART_On{jj,1};
    O_sample = O_tmp(:,1:5:end);
    
    FT_tmp = EE_FTn{jj,1};
    FT_sample = FT_tmp(:,1:5:end);
    
    H_tmp = EE_Hn{jj,1};
    H_sample = H_tmp(:,:,1:5:end);
    
    time_P{jj,1} = time_sample;
    EE_PnP{jj,1} = P_sample;
    EE_OnP{jj,1} = O_sample;
    EE_FTnP{jj,1} = FT_sample;
    EE_HnP{jj,1} = H_sample;
end

%% Convert to 6D for Segmentation
Xn = [];
for ii=1:length(EE_Hn)    
    %From pos + or
%     X = cat(1,EE_CART_Pn{ii,1},EE_CART_On{ii,1});
    X = cat(1,EE_CART_Pn{ii,1},EE_CART_On{ii,1}/3.1416);
    Xn{ii,1} = X;
end

%% Convert to 7D for Segmentation
XnQ = {};
HnQ = {};
for ii=1:length(EE_Hn)
    H = EE_HnP{ii,1};
    R = H(1:3,1:3,:);
    t = reshape(H(1:3,4,:),3,length(H));
    q = quaternion(R);
    Xq = cat(1,t,q);    
    XnQ{ii,1} = Xq;
end


%% Check Rotation
Xn_ch = [];
HnQ = {};
for jj=1:length(XnQ)
    %Quaternion
    X_test = XnQ{jj,1}; rot_end = 7;
    %Angle
%     X_test = Xn{jj,1};rot_end = 6;
    X_new = X_test;
%     for i=4:1:rot_end
%         X_new(i,:) = checkRotation(X_test(i,:));
%     end    
    Q_new = checkRotations(X_test(4:rot_end,:));
    X_new = cat(1,X_test(1:3,:),Q_new);
    Xn_ch{jj,1} = X_new;
    figure;
    subplot(2,1,1)
    plot(X_test')
    legend('x','y','z','qw','qx','qy','qz')

    subplot(2,1,2)
    plot(X_new')
    
    % Convert to H to check rotations
    Rq = quaternion(Q_new);
    tq = EE_HnP{jj,1}(1:3,4,:);
    Hq = zeros(4,4,length(Rq));
    for ii=1:length(Rq)
       Hii =  eye(4);
       Hii(1:3,1:3) = Rq(:,:,ii);
       Hii(1:3,4) = tq(:,:,ii);
       Hq(:,:,ii) = Hii;  
    end
    HnQ{jj,1} = Hq; 
end

%% Smoothen FT Sensor Data
Smooth_FTn ={};
for jj=1:length(Sensor_FTn)
    sm_ft = Sensor_FTnP{jj,1};
    for kk=1:size(sm_ft,1)
        sm_ft(kk,:) = smooth(sm_ft(kk,:),0.01,'moving');
    end
    Smooth_FTn{jj,1} = sm_ft;
end

%% Visualize Trajectories + FT
figure('Color',[1 1 1])
n = 1;
subplot(2,1,1)
plot(Xn_ch{n,1}(1:3,:)')
legend('x','y','z')
subplot(2,1,2)
plot(Xn_ch{n,1}(4:end,:)')
legend('qw','qx','qy','qz')


%% Visualize Rigid Body Trajectories
t= 1;
named_figure('Sauce Pouring Trial Raw ', t);
clf;
% drawframetraj(EE_HnP{t},0.005,1);
drawframetraj(EE_HnP{t}(:,:,464:778),0.005,1);


%% Convert to 7D for Segmentation
Xn = [];
for ii=1:length(EE_Hn)       
    %From pos + or (quaternion)
    X = Xn_ch{ii,1};
    Xn{ii,1} = X;
end

%% Preprocess for segmentation (From MoCap data pre-processing experiment)
 
RobotChannelNames = {'pos.x','pos,y','pos.z', 'pos.qw','pos.qi','pos.qj','pos.qz'};
% RobotChannelNames = {'pos.x','pos,y','pos.z', 'pos.roll','pos.pitch','pos.yaw'};


N = length(Xn);
% N = 5;
nDim = 7;
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
start_demo = [166, 194,232,115, 153,161,96, 122, 96,87, 159,134,108,96,15];
end_demo = [722, 932,922, 781, 835, 812, 813,697,703,710,758,673,733,770, 863];

EE_PnPSeg = [];
EE_HnPSeg = [];
HnQSeg = [];

% Read in every sequence, one at a time
for ii = 1:N
    % Pos + Or 
    D = Xn{ii}(4:7,start_demo(ii):end_demo(ii));

    EE_PnPSeg{ii,1} = Xn{ii}(:,start_demo(ii):end_demo(ii));
    EE_HnPSeg{ii,1} = EE_HnP{ii}(:,:,start_demo(ii):end_demo(ii));
    HnQSeg{ii,1} = HnQ{ii}(:,:,start_demo(ii):end_demo(ii));

%     D = Xn{ii};
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

%% Visualize Trajectories + FT
figure('Color',[1 1 1])
n = 15;
subplot(2,1,1)
plot(EE_PnPSeg{n,1}(1:3,:)')
legend('x','y','z')
subplot(2,1,2)
plot(EE_PnPSeg{n,1}(4:end,:)')
legend('qw','qx','qy','qz')

%% Visualize Rigid Body Trajectories
t= 1;
named_figure('Sauce Pouring Trial Raw ', t);
clf;
% drawframetraj(EE_HnPSeg{t},0.005,1);
drawframetraj(EE_HnPSeg{t}(:,:,286:557),0.005,1);

%%
Segm_results_me = [];
Segm_results_me{1,1} = [1 286; 2 length(EE_PnPSeg{1})];
Segm_results_me{2,1} = [1 305; 2 length(EE_PnPSeg{2})];
Segm_results_me{3,1} = [1 317; 2 length(EE_PnPSeg{3})];
Segm_results_me{4,1} = [1 240; 2 length(EE_PnPSeg{4})];
Segm_results_me{5,1} = [1 350; 2 length(EE_PnPSeg{5})];
Segm_results_me{6,1} = [1 274; 2 length(EE_PnPSeg{6})];
Segm_results_me{7,1} = [1 355; 2 length(EE_PnPSeg{7})];
Segm_results_me{8,1} = [1 256; 2 length(EE_PnPSeg{8})];
Segm_results_me{9,1} = [1 288; 2 length(EE_PnPSeg{9})];
Segm_results_me{10,1} = [1 314; 2 length(EE_PnPSeg{10})];
Segm_results_me{11,1} = [1 308; 2 length(EE_PnPSeg{11})];
Segm_results_me{12,1} = [1 243; 2 length(EE_PnPSeg{12})];
Segm_results_me{13,1} = [1 328; 2 length(EE_PnPSeg{13})];
Segm_results_me{14,1} = [1 394; 2 length(EE_PnPSeg{14})];
Segm_results_me{15,1} = [1 539; 2 length(EE_PnPSeg{15})];

%% Plot Segmented Trajectories and Construct Structures
seq = [9];
Origin = eye(4);

for i=1:max(Total_feats)
    color(i,:) = [rand rand rand];
end

% Kinect Origin from LASA Perception Module
% - Translation: [0.244, -0.364, 1.246]
% - Rotation: in Quaternion [0.718, 0.666, -0.146, -0.140]
%             in RPY [-2.735, 0.023, 1.501]
Camera_Origin = eye(4);
R = quaternion([ -0.140, 0.718, 0.666, -0.146]);
t = [0.244, -0.364, 1.246];
Camera_Origin = eye(4);
Camera_Origin(1:3,1:3) = R;
Camera_Origin(1:3,4) = t;

% Dough Frame from LASA Perception Module
% - Translation: [-0.549, -0.328, -0.044]
% - Rotation: in Quaternion [-0.000, -0.000, 0.999, 0.037]
%             in RPY [-0.000, 0.000, 3.068]

R = quaternion([  0.037, -0.000, -0.000, 0.999]);
t = [-0.549, -0.328, -0.044];
Dough_Frame = eye(4);
Dough_Frame(1:3,1:3) = R;
Dough_Frame(1:3,4) = t;
Dough_Area = 0.0403079;

table_off = 0.02;
table_width = 0.56;
table_height = 0.75;

% Phase1_3rd = Phase1;
% Phase2_3rd = Phase2;
% Phase3_3rd = Phase3;

Phase1 = [];
Phase2 = [];
Phase3 = [];


for j=1:length(seq)
    figure('Color',[1 1 1])
    %%%%%Table Localization%%%%%
    Table_Origin = eye(4);
    Table_Origin(:,4) = [-0.2-table_height 0 -0.044-0.02 1] ;
    Table_Edge1=eye(4);Table_Edge2=eye(4);Table_Edge3=eye(4);Table_Edge4=eye(4);
    Table_Edge1(1:3,4) = [-table_off 0 table_off]'; Table_Edge1 = Table_Origin*Table_Edge1;
    Table_Edge2(1:3,4) = [0 -table_width 0]'; Table_Edge2 = Table_Edge1*Table_Edge2;
    Table_Edge3(1:3,4) = [table_height -table_width 0]'; Table_Edge3 = Table_Edge1*Table_Edge3;
    Table_Edge4(1:3,4) = [table_height 0 0]'; Table_Edge4 = Table_Edge1*Table_Edge4;
    
    TrajSeg = [];
    TrajSeg = Segm_results_me{seq(j),1};
    Traj = EE_PnPSeg{seq(j),1};
    SegPoints = [1; TrajSeg(:,2)]

    for i=1:length(TrajSeg)        
            start_seg = SegPoints(i);
            if i<length(TrajSeg)
                end_seg = SegPoints(i+1);
            else
                end_seg = length(Traj);
            end
            
            X = Traj(1,start_seg:end_seg);
            Y = Traj(2,start_seg:end_seg); 
            Z = Traj(3,start_seg:end_seg);               
          
            drawframe(Dough_Frame,0.1)          
            hold on 
            
            plot3(X,Y,Z,'Color', color(TrajSeg(i,1),:),'LineWidth', 3);
            hold on
            text(Traj(1,SegPoints(i)),Traj(2,SegPoints(i)),Traj(3,SegPoints(i))+0.05,num2str(TrajSeg(i,1)),'FontSize',16, 'Color', color(TrajSeg(i,1),:));
            drawframe(EE_HnPSeg{seq(j),1}(:,:,SegPoints(i)),0.03)
            drawframe(EE_HnPSeg{seq(j),1}(:,:,SegPoints(i+1)),0.03)
            
            for k=SegPoints(i):20:SegPoints(i+1)
                EE_frame_k = EE_HnPSeg{seq(j),1}(:,:,k);
                drawframe(EE_frame_k,0.03)
            end
            
            
            % Make Data Structure for Lucia
            clear action
            if (TrajSeg(i,1)==1)
                action.description = 'pour';    
            end
            if (TrajSeg(i,1)==2)
                action.description = 'back';    
            end
            
            action.BP_HMM_id = TrajSeg(i,1);
            action.sampling_ratio = 5;
            action.dough_area = Dough_Area; 
            action.dough_frame_in_world = Dough_Frame;
            
            action.in_world.EE_Hfull =  HnQSeg{seq(j),1}(:,:,start_seg:end_seg);
            
            EE_Hfull_in_D = zeros(size(action.in_world.EE_Hfull));
            
            for kk=1:size(action.in_world.EE_Hfull,3)
                EE_in_W = action.in_world.EE_Hfull(:,:,kk);
                EE_in_D = (action.dough_frame_in_world^-1)*EE_in_W;
                EE_Hfull_in_D(:,:,kk) = EE_in_D;               
            end           
            
            % EE stuff
            H = EE_Hfull_in_D;
            R = H(1:3,1:3,:);
            t = reshape(H(1:3,4,:),3,length(H));
            q = quaternion(R,true);
               
            action.in_dough.EE_POS = t;
            action.in_dough.EE_ORI = q;
            action.in_dough.EE_Hfull =  EE_Hfull_in_D;
            
            action.in_dough_interp.EE_POS = [];
            action.in_dough_interp.EE_ORI = [];
            
            if (action.BP_HMM_id == 1)
                Phase1 = [Phase1 action];
            end
            if (action.BP_HMM_id == 2)
                Phase2 = [Phase2 action];
            end                       
    end
   
    % Plot Beginning and End of Recording and Camera Origin
    scatter3(Traj(1,1),Traj(2,1),Traj(3,1), 100, [0 1 0], 'filled')
    scatter3(Traj(1,end),Traj(2,end),Traj(3,end),100, [1 0 0], 'filled')
    scatter3(Camera_Origin(1,4),Camera_Origin(2,4),Camera_Origin(3,4),50, [0 0 0], 'filled')
    drawframe(Camera_Origin,0.1)
    drawframe(Origin,0.1)
    
    % Draw Table
    fill3([Table_Edge1(1,4) Table_Edge2(1,4) Table_Edge3(1,4) Table_Edge4(1,4)],[Table_Edge1(2,4) Table_Edge2(2,4) Table_Edge3(2,4) Table_Edge4(2,4)],[Table_Edge1(3,4) Table_Edge2(3,4) Table_Edge3(3,4) Table_Edge4(3,4)],[0.5 0.5 0.5])    
        
    str = strcat('Sequence ',num2str(seq(j)));
    title(str);
    xlabel('x');ylabel('y');zlabel('z');
    axis equal
    grid on
end


%% Check transformed data

%In World
figure('Color',[1 1 1])

i=6;
plot3(reshape(Phase1(i).in_world.EE_Hfull(1,4,:),[1 length((Phase1(i).in_world.EE_Hfull(1,4,:)))]),reshape(Phase1(i).in_world.EE_Hfull(2,4,:),[1 length((Phase1(i).in_world.EE_Hfull(2,4,:)))]),reshape(Phase1(i).in_world.EE_Hfull(3,4,:),[1 length((Phase1(i).in_world.EE_Hfull(3,4,:)))]),'Color', [rand rand rand] ,'LineWidth', 3);    
hold on
plot3(reshape(Phase2(i).in_world.EE_Hfull(1,4,:),[1 length((Phase2(i).in_world.EE_Hfull(1,4,:)))]),reshape(Phase2(i).in_world.EE_Hfull(2,4,:),[1 length((Phase2(i).in_world.EE_Hfull(2,4,:)))]),reshape(Phase2(i).in_world.EE_Hfull(3,4,:),[1 length((Phase2(i).in_world.EE_Hfull(3,4,:)))]),'Color', [rand rand rand] ,'LineWidth', 3);    
hold on
plot3(reshape(Phase3(i).in_world.EE_Hfull(1,4,:),[1 length((Phase3(i).in_world.EE_Hfull(1,4,:)))]),reshape(Phase3(i).in_world.EE_Hfull(2,4,:),[1 length((Phase3(i).in_world.EE_Hfull(2,4,:)))]),reshape(Phase3(i).in_world.EE_Hfull(3,4,:),[1 length((Phase3(i).in_world.EE_Hfull(3,4,:)))]),'Color', [rand rand rand] ,'LineWidth', 3);    
hold on

drawframe(Phase1(i).in_world.EE_Hfull(:,:,1),0.03)
drawframe(Phase2(i).in_world.EE_Hfull(:,:,1),0.03)
drawframe(Phase3(i).in_world.EE_Hfull(:,:,1),0.03)
drawframe(eye(4),0.05)

axis equal 
grid on

%roll
figure('Color',[1 1 1])
subplot(2,1,1)
plot(Phase2(i).in_world.Sensor_FOR'); legend('fx','fy','fz')
subplot(2,1,2)
plot(Phase2(i).in_world.Sensor_TQS'); legend('taux','tauy','tauz')

%In Dough
figure('Color',[1 1 1])
plot3(reshape(Phase1(i).in_dough.EE_Hfull(1,4,:),[1 length((Phase1(i).in_dough.EE_Hfull(1,4,:)))]),reshape(Phase1(i).in_dough.EE_Hfull(2,4,:),[1 length((Phase1(i).in_dough.EE_Hfull(2,4,:)))]),reshape(Phase1(i).in_dough.EE_Hfull(3,4,:),[1 length((Phase1(i).in_dough.EE_Hfull(3,4,:)))]),'Color', [rand rand rand] ,'LineWidth', 3);    
% plot3(reshape(Phase1(i).in_dough.EE_POS(1,:),[1 length((Phase1(i).in_dough.EE_POS(1,:)))]),reshape(Phase1(i).in_dough.EE_POS(2,:),[1 length((Phase1(i).in_dough.EE_POS(2,:)))]),reshape(Phase1(i).in_dough.EE_POS(3,:),[1 length((Phase1(i).in_dough.EE_POS(3,:)))]),'Color', [rand rand rand] ,'LineWidth', 3);    
hold on
plot3(reshape(Phase2(i).in_dough.EE_Hfull(1,4,:),[1 length((Phase2(i).in_dough.EE_Hfull(1,4,:)))]),reshape(Phase2(i).in_dough.EE_Hfull(2,4,:),[1 length((Phase2(i).in_dough.EE_Hfull(2,4,:)))]),reshape(Phase2(i).in_dough.EE_Hfull(3,4,:),[1 length((Phase2(i).in_dough.EE_Hfull(3,4,:)))]),'Color', [rand rand rand] ,'LineWidth', 3);    
% plot3(reshape(Phase2(i).in_dough.EE_POS(1,:),[1 length((Phase2(i).in_dough.EE_POS(1,:)))]),reshape(Phase2(i).in_dough.EE_POS(2,:),[1 length((Phase2(i).in_dough.EE_POS(2,:)))]),reshape(Phase2(i).in_dough.EE_POS(3,:),[1 length((Phase2(i).in_dough.EE_POS(3,:)))]),'Color', [rand rand rand] ,'LineWidth', 3);    
hold on
plot3(reshape(Phase3(i).in_dough.EE_Hfull(1,4,:),[1 length((Phase3(i).in_dough.EE_Hfull(1,4,:)))]),reshape(Phase3(i).in_dough.EE_Hfull(2,4,:),[1 length((Phase3(i).in_dough.EE_Hfull(2,4,:)))]),reshape(Phase3(i).in_dough.EE_Hfull(3,4,:),[1 length((Phase3(i).in_dough.EE_Hfull(3,4,:)))]),'Color', [rand rand rand] ,'LineWidth', 3);    
% plot3(reshape(Phase3(i).in_dough.EE_POS(1,:),[1 length((Phase3(i).in_dough.EE_POS(1,:)))]),reshape(Phase3(i).in_dough.EE_POS(2,:),[1 length((Phase3(i).in_dough.EE_POS(2,:)))]),reshape(Phase3(i).in_dough.EE_POS(3,:),[1 length((Phase3(i).in_dough.EE_POS(3,:)))]),'Color', [rand rand rand] ,'LineWidth', 3);    
hold on

drawframe(Phase1(i).in_dough.EE_Hfull(:,:,1),0.03)
drawframe(Phase2(i).in_dough.EE_Hfull(:,:,1),0.03)
drawframe(Phase3(i).in_dough.EE_Hfull(:,:,1),0.03)
drawframe(eye(4),0.05)
axis equal 
grid on

%roll
figure('Color',[1 1 1])
subplot(2,1,1)
plot(Phase2(i).in_dough.Sensor_FOR'); legend('fx','fy','fz')
subplot(2,1,2)
plot(Phase2(i).in_dough.Sensor_TQS'); legend('taux','tauy','tauz')

%% PLot all phases to check for consistency
% Phase1 = Phase1_all;
% Phase2 = Phase2_all;
% Phase3 = Phase3_all;

figure('Color',[1 1 1])
title('Phase 1 (reach)')
for i = 1:length(Phase1)
% for i = 13:18
    xyz = Phase1(i).in_dough.EE_POS;
    plot3(xyz(1,:),xyz(2,:),xyz(3,:),'Color', [rand rand rand] ,'LineWidth', 2);   
    hold on
end
drawframe(eye(4),0.05)
axis equal 
grid on

figure('Color',[1 1 1])
title('Phase 2 (pour)')
for i = 1:length(Phase1)
% for i = 13:18    
    xyz = Phase2(i).in_dough.EE_POS;
    plot3(xyz(1,:),xyz(2,:),xyz(3,:),'Color', [rand rand rand] ,'LineWidth', 2);   
    drawframe(Phase1(i).in_dough.EE_Hfull(:,:,end),0.03)
    hold on
end
drawframe(eye(4),0.05)
axis equal 
grid on

%% Interpolation station
%---- PHASE 1 ---%
min_lengths = length(Phase1(1).in_dough.EE_POS);
for ii=1:length(Phase1)   
        min_lengths = [min_lengths length(Phase1(ii).in_dough.EE_POS)];
end 
mean_length = ceil(mean(min_lengths));
min_length = min(min_lengths);
min_length = mean_length;
k = 1;
for jj=1:15
    figure('Color',[1 1 1])
    for ii=1:5
        subplot(5,1,ii)
        Phase = Phase1(k); 
        POS = Phase.in_dough.EE_POS;
        New_POS = interpolateSpline(POS,min_length);

        ORI = Phase.in_dough.EE_ORI;
        New_ORI = interpolateSpline(ORI,min_length);
               
        Interp_Data = [New_POS; New_ORI];
        plot(Interp_Data')
        name=strcat('Sequence',num2str(k))
        title(name)
        Phase1(k).in_dough_interp.EE_POS = New_POS;
        Phase1(k).in_dough_interp.EE_ORI = New_ORI;
        k=k+1;
    end
end
%% Interpolation station
%---- PHASE 2 ---%
min_lengths = length(Phase2(1).in_dough.EE_POS);
for ii=1:length(Phase2)   
        min_lengths = [min_lengths length(Phase2(ii).in_dough.EE_POS)];
end 
mean_length = ceil(mean(min_lengths));
min_length = mean_length;
k = 1;
for jj=1:18
    figure('Color',[1 1 1])
    for ii=1:5
        subplot(5,1,ii)
        Phase = Phase2(k); 
        POS = Phase.in_dough.EE_POS;
        New_POS = interpolateSpline(POS,min_length);

        ORI = Phase.in_dough.EE_ORI;
        New_ORI = interpolateSpline(ORI,min_length);      
        
        Interp_Data = [New_POS; New_ORI];
        plot(Interp_Data')
        name=strcat('Sequence',num2str(k))
        title(name)
        Phase2(k).in_dough_interp.EE_POS = New_POS;
        Phase2(k).in_dough_interp.EE_ORI = New_ORI;
        k=k+1;
    end
end

%% PLot all phases after interpol to check for consistency
figure('Color',[1 1 1])
title('Phase 1 (pour)')

%To compute attractors
Home = eye(4);
t_home = [];

Attr_Phase1 = eye(4);
t_Phase1 = [];
% for i = 1:length(Phase1)
for i=5:9
% for i = 13:18
    
    xyz = Phase1(i).in_dough_interp.EE_POS;
    plot3(xyz(1,:),xyz(2,:),xyz(3,:),'Color', [rand rand rand] ,'LineWidth', 2);   
    drawframe(Phase1(i).in_dough.EE_Hfull(:,:,1),0.02)
    drawframe(Phase1(i).in_dough.EE_Hfull(:,:,end),0.02)
%     t_home = [t_home xyz(:,1)];
%     t_Phase1 = [t_Phase1 xyz(:,end)];
%     
    t_home = [xyz(:,1)];
    t_Phase1 = [xyz(:,end)];
    hold on
end
t_Phase1 = sum(t_Phase1,2)
Attr_Phase1(1:3,1:3) = Phase1(i).in_dough.EE_Hfull(1:3,1:3,end);
Attr_Phase1(1:3,4) = t_Phase1;
q_Phase1 = quaternion(Attr_Phase1(1:3,1:3))
drawframe(Attr_Phase1,0.07)

t_home = sum(t_home,2)
Home(1:3,1:3) = Phase1(i).in_dough.EE_Hfull(1:3,1:3,1);
Home(1:3,4) = t_home;
q_home = quaternion(Home(1:3,1:3))

% Home_w = ((Home^-1)*(Dough_Frame^-1))^-1;
% t_home_w = Home_w(1:3,4)
% q_HOME = quaternion(Home_w(1:3,1:3))
drawframe(Home,0.07)


drawframe(eye(4),0.05)
axis equal 
grid on

figure('Color',[1 1 1])
title('Phase 2 (back)')

Attr_Phase2 = eye(4);
t_Phase2 = [];

% for i = 1:length(Phase1)
for i = 10:15 
    xyz = Phase2(i).in_dough_interp.EE_POS;
    plot3(xyz(1,:),xyz(2,:),xyz(3,:),'Color', [rand rand rand] ,'LineWidth', 2);   
    drawframe(Phase2(i).in_dough.EE_Hfull(:,:,end),0.02)
    t_Phase2 = [xyz(:,end)];
    hold on
end

t_Phase2 = sum(t_Phase2,2)
Attr_Phase2(1:3,1:3) = Phase2(i).in_dough.EE_Hfull(1:3,1:3,end);
Attr_Phase2(1:3,4) = t_Phase2;
q_Phase2 = quaternion(Attr_Phase2(1:3,1:3))

drawframe(Attr_Phase2,0.07)
drawframe(eye(4),0.05)
axis equal 
grid on

%% Save Segmentation Data
save('Pouring4Joel_All_Trials_Segmented.mat','Phase1','Phase2')

%% Make text files for Master/slave models 

% write to a new file
fid=fopen('slaveGMM.txt','w');
fprintf(fid, '%f\n', size(structGMM.Mu,2));
fprintf(fid, '%f\n\n', size(structGMM.Mu,1));

fprintf(fid, '%f\n\n', structGMM.States);

for i=1:length(structGMM.Mu)
    fprintf(fid, '%f\n', structGMM.Mu(i));
end

for i=1:size(structGMM.Sigma,1)
    fprintf(fid, '\n');
    for j=1:size(structGMM.Sigma,2)
        fprintf(fid, '%f\t ', structGMM.Sigma(i,j));
    end
end

fclose(fid);

%% Make text files for CPL models
fid=fopen('clpGMM.txt','w');
fprintf(fid, '%f\n', size(structGMM.Mu,2));
fprintf(fid, '%f\n\n', size(structGMM.Mu,1));

for i=1:length(structGMM.Priors)
   fprintf(fid, '%f\t ', structGMM.Priors(i));
end
fprintf(fid, '\n');
    
for i=1:size(structGMM.Mu,1)
    fprintf(fid, '\n');
    for j=1:size(structGMM.Mu,2)
        fprintf(fid, '%f\t ', structGMM.Mu(i,j));
    end
end


for i=1:size(structGMM.Sigma,3)
    fprintf(fid, '\n');
    for j=1:size(structGMM.Sigma,1)
        fprintf(fid, '\n');
        for k=1:size(structGMM.Sigma,2)
            fprintf(fid, '%f\t ', structGMM.Sigma(j,k,i));
        end
    end
end

fclose(fid);

%% Back Attractor
%Sent Position: -0.484177,0.0843954,0.322472
%Sent Orientation: 0.736958,-0.515257,0.18616,0.395915

Back_W = eye(4);
Back_W(1:3,4) = [-0.484177,0.0843954,0.322472];
Back_W(1:3,1:3) = quaternion([0.395915, 0.736958,-0.515257,0.18616]);
Back_D = ((Back_W^-1)*Dough_Frame)^-1;
back_d_t = Back_D(1:3,4)
back_d_q = quaternion(Back_D(1:3,1:3))