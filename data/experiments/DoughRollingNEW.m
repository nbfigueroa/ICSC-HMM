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
    X = cat(1,EE_CART_Pn{ii,1},EE_CART_On{ii,1});
%     X = cat(1,EE_CART_Pn{ii,1},EE_CART_On{ii,1}/3.1416);
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
    
    
    % Smoothen FT Sensor Data
        sm_ft = FT;
        for kk=1:size(sm_ft,1)
            sm_ft(kk,:) = smooth(sm_ft(kk,:),0.01,'moving');
        end
    X_tmp = cat(1,X,sm_ft);  
%     X_tmp = cat(1,X,FT);  
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
%     for i=4:1:rot_end
%         X_new(i,:) = checkRotation(X_test(i,:));
%     end
    
    Q_new = checkRotations(X_test(4:rot_end,:))*1.5;
    X_tmp = cat(1,X_test(1:3,:),Q_new);
    X_new = cat(1,X_tmp,X_test(rot_end+1:end,:));
    Xn_ch{jj,1} = X_new;
    
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
% subplot(2,1,1)
plot(Xn_ch{n,1}')
% legend('x','y','z','roll','pitch','yaw')
% subplot(2,1,2)
% plot(EE_FTn{n,1}')
% legend('fx','fy','fz','tx','ty','tz')
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

%% Visualize Trajectories + FT
figure('Color',[1 1 1])
n = 4;
subplot(4,1,1)
plot(Xn_ch{n,1}(1:3,:)')
legend('x','y','z')
subplot(4,1,2)
plot((Xn_ch{n,1}(4:6,:)*(3.1416/1.5))')
legend('roll','pitch','yaw')
% subplot(2,1,2)
% plot(EE_FTn{n,1}')
% legend('fx','fy','fz','tx','ty','tz')
% subplot(2,1,2)
% plot(Sensor_FTnP{n,1}') 
% legend('fx','fy','fz','tx','ty','tz')
subplot(4,1,3)
plot((Xn_ch{n,1}(7:9,:)*10)') 
legend('fx','fy','fz')

subplot(4,1,4)
plot((Xn_ch{n,1}(10:end,:)*10)') 
legend('tx','ty','tz')


%% Preprocess for segmentation (From MoCap data pre-processing experiment)
 
RobotChannelNames = {'pos.x','pos,y','pos.z', 'pos.qw','pos.qi','pos.qj','pos.qz'};
% RobotChannelNames = {'pos.x','pos,y','pos.z', 'pos.roll','pos.pitch','pos.yaw'};


% Visualize 3d trajectories different color is different timeseries
load('./code/demo/demo_data/rolling_same_orientation.mat')
XnT = Xn_ch(2:end);

N = length(XnT);
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


figure('Color', [1 1 1])
% Read in every sequence, one at a time
for ii = 1:N

    % Pos + Or + F + T
    D = XnT{ii}(:,:);  
    scatter3(D(1,:),D(2,:),D(3,:),5, [rand rand rand],'filled')
    hold on
    
    % Enforce zero-mean
    D = D';
    D = bsxfun( @minus, D, mean(D,1) );
    D = D';
    
    % Add the sequence to our data structure
    Robotdata = Robotdata.addSeq( D, num2str(ii));
    data_struct(ii).obs = D;
%     data_struct(ii).true_labels = Truelabels;
end

grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z')
% -------------------------------------- Preproc autoregressive data
% This step simply builds necessary data structs for efficient AR inference
%   including filling in XprevR data field so that for any time tt,
%       XprevR(:,tt) = [Xdata(:, tt-1); Xdata(:,tt-2); ... Xdata(:,tt-R) ]
%   allowing of course for discontinuities at sequence transitions
RobotdataAR = ARSeqData( Preproc.R, Robotdata);

%% Load Dough Position and Principal Direction
dd = './'
% d = './';
% Series #
subdird = strcat(dd,'series_3_area/');
subdird = strcat(dd,'series_2_area/');

filesd = dir(strcat(subdird,'*.txt'));
dough = zeros(length(filesd),3);
dough_d1 = zeros(length(filesd),2);
for ii=1:length(filesd)
    filename = strcat(subdird,filesd(ii,1).name);    
    
    fileID = fopen(filename);
% {"Dough_D1": [232, 156], "Dough_D2": [498, 141], "Dough_Center": [377, 230], "ID": 1, "Area": 7727.0}
    C = textscan(fileID,'%s[%d,%d],%s[%d,%d],%s[%d,%d],%s%d,%s%f');
    fclose(fileID);
    dough(C{11}+1,:) = [C{8} C{9} ceil(C{13})];
    dough_d1(C{11}+1,:) = [C{2} C{3}];
end
dough(1,:) =[]
dough_d1(1,:) =[]
%% Clean up oversegmentations and Plot Segmented Trajectory
desired_action_sequence = [3 1 2];
num_atomic_seqs = [];
Segm_results_clean = {};

%Series2
bad_seq = 3;

clear diff
for i=1:length(Segm_results)
    display('here')
    if (i==bad_seq)
        Segm_results_clean{i,1} = [];
        num_atomic_seqs = [num_atomic_seqs; 3]; 
    else

        Segments = Segm_results{i};
    %     if ~(i==bad_seq)
        Traj = Xn_ch{i,1};
        SegPoints_beg = [1; Segments(:,2)];
        SegPoints_end = [Segments(:,2); length(Traj)];
        Seg_length = SegPoints_beg-SegPoints_end;
        Segm_results_new = Segments;
        for j=1:(length(Seg_length))
            if abs(Seg_length(j)) < 40
               Segm_results_new(j,1) = Segm_results_new(j-1,1); 
            end
        end 

        if Segm_results_new(end,2)==0
            Segm_results_new(end,:) = []; 
        end

    %     %Series 3
    %     if ~isempty(find(Segm_results_new(:,1)==4))        
    %         Segm_results_new(find(Segm_results_new(:,1)==4),1) = 1;
    %     end

        %Series 2
        if ~isempty(find(Segm_results_new(:,1)==4))        
            Segm_results_new(find(Segm_results_new(:,1)==4),1) = 3;
        end

        if ~isempty(find(Segm_results_new(:,1)==5))        
            Segm_results_new(find(Segm_results_new(:,1)==5),1) = 1;
        end

        if Segm_results_new(end,1)~=2
            Segm_results_new(end,:) = []; 
        end

        Segm_results_merged = [];
        s = diff(Segm_results_new(:,1));   
        if isempty(find(s==0))
            Segm_results_merged = Segm_results_new;    
        else
            iter = 0;
            while (~isempty(find(s==0)))
                if iter>0
                    tmp = Segm_results_merged;
                    Segm_results_new = Segm_results_merged;
                    Segm_results_merged = [];
                end
                j = 1;
                origin_ids = 1:length(Segm_results_new);
                 while(j < length(origin_ids)+1)
                     if j == length(origin_ids)
                        Segment = Segm_results_new(origin_ids(j),:);
                        Segm_results_merged = [Segm_results_merged; Segment];          
                     else
                        if Segm_results_new(origin_ids(j)) == Segm_results_new(origin_ids(j+1))
                            Segment = Segm_results_new(origin_ids(j+1),:);
                            Segm_results_merged = [Segm_results_merged; Segment];
                            origin_ids(j+1) = [];                         
                        else
                            Segment = Segm_results_new(origin_ids(j),:);
                            Segm_results_merged = [Segm_results_merged; Segment];          
                        end
                     end
                    j = j+1;
                    iter = iter + 1;
                 end
                 s = diff(Segm_results_merged(:,1));
            end
        end

        if Segm_results_merged(1,1)~=3
            Segm_results_merged(1,:) =[];
        end

        tmp = Segm_results_merged;
        for ii=1:length(tmp)
            if (Segm_results_merged(ii,1)~=1) && (Segm_results_merged(ii,1)~=2) && (Segm_results_merged(ii,1)~=3)
                tmp(ii,:) =[];
            end
        end

        Segm_results_merged = tmp
        Segm_results_clean{i,1} = Segm_results_merged;
        num_atomic_seqs = [num_atomic_seqs; length(find(Segm_results_merged(:,1)==1))];
    end
%     if (i==bad_seq)
%         Segm_results_clean{i,1} = [];
%     end
%     end
end

dough_seq_id = [];
for i=1:length(num_atomic_seqs)
    if i==1
        dough_start = 1;
    else        
        dough_start = dough_seq_id(i-1,2)+1;
    end   
    dough_end = dough_start + num_atomic_seqs(i);
    dough_seq_id = [dough_seq_id; dough_start dough_end];
end


%% Sequence Hack for Series 3
% for ii=1:length(Segm_results_clean)
%     tmp = Segm_results_clean{ii};
%     if length(tmp)>9
%         tmp(4:5,:) = [];
%     end
%     Segm_results_clean{ii} = tmp;
% end

%% Sequence Hack for Series 2
for i=2:2
tmp = Segm_results_clean{i};
tmp(4:5,:) = [];
Segm_results_clean{i} = tmp;
end

for i=4:4
tmp = Segm_results_clean{i};
tmp(3,:) = [];
Segm_results_clean{i} = tmp;
end

for i=5:5
tmp = Segm_results_clean{i};
tmp(8:9,:) = [];
Segm_results_clean{i} = tmp;
end

for i=6:6
tmp = Segm_results_clean{i};
tmp(9,:) = [];
Segm_results_clean{i} = tmp;
end

for i=7:7
tmp = Segm_results_clean{i};
tmp([1:2 5],:) = [];
Segm_results_clean{i} = tmp;
end

for i=8:8
tmp = Segm_results_clean{i};
tmp(6,:) = [];
Segm_results_clean{i} = tmp;
end


for i=15:15
tmp = Segm_results_clean{i};
tmp(5:6,:) = [];
Segm_results_clean{i} = tmp;
end

%% Plot Segmented Trajectories and Construct Structures
seq = [12];
%Until 7 it's safe
% seq = [1:2 4:15];
Origin = eye(4);

for i=1:max(Total_feats)
    color(i,:) = [rand rand rand];
end

table_off = 0.02;
table_width = 0.56;
table_height = 0.75;

% Series 3 Details
% Chess_width_left = -0.075;
% Chess_height_top = 0.013;
% o = [556 14]; o_t = normalizeNadia(o',fc,cc,kc,alpha_c)*Tc_ext(3)*0.001;

% Series 2 Details
Chess_width_left = -0.075
Chess_height_top = 0.031
o = [505 7]; o_t = normalizeNadia(o',fc,cc,kc,alpha_c)*Tc_ext(3)*0.001;

Phase1 = [];
Phase2 = [];
Phase3 = [];

for j=1:length(seq)
    figure('Color',[1 1 1])
    %%%%%Table Localization%%%%%
    Table_Origin = Table_Hn{seq(j),1}(:,:,1);
    Table_Edge1 = eye(4); Table_Edge2 = eye(4); Table_Edge3 = eye(4); Table_Edge4 = eye(4);
    Table_Edge1(1:3,4) = [-table_off 0 table_off]'; Table_Edge1 = Table_Origin*Table_Edge1;
    Table_Edge2(1:3,4) = [0 0 -table_width]'; Table_Edge2 = Table_Edge1*Table_Edge2;
    Table_Edge3(1:3,4) = [table_height 0 -table_width]'; Table_Edge3 = Table_Edge1*Table_Edge3;
    Table_Edge4(1:3,4) = [table_height 0 0]'; Table_Edge4 = Table_Edge1*Table_Edge4;
    
    %%%%%Chess Origin Localization%%%%%
    Chess_Origin = eye(4); Chess_Origin(1:3,1:3) = rotx(-pi/2); Chess_Origin(1:3,4) = [Chess_height_top 0 -(table_width+Chess_width_left)]'; Chess_Origin = Table_Edge1*Chess_Origin;
    Chess_Origin_In_Camera = eye(4); Chess_Origin_In_Camera(1:3,1:3) = Rc_ext; Chess_Origin_In_Camera(1:3,4) = ([Tc_ext(1)*0.001 Tc_ext(2)*0.001 0])'; 
    
    %%%%%Camera Localization%%%%%
    Camera_Origin_on_Table = (Chess_Origin_In_Camera*Chess_Origin^-1)^-1;
    Chess_Origin_In_Camera(1:3,4) = ([Tc_ext(1)*0.001 Tc_ext(2)*0.001 Tc_ext(3)*0.001])';
    Camera_Origin_In_World = (Chess_Origin_In_Camera*Chess_Origin^-1)^-1;
 
    %%%%%Dough Localization%%%%%
    dough_start = dough_seq_id(seq(j),1);
    dough_end = dough_seq_id(seq(j),2);
    
    dough_seq = dough(dough_start:dough_end,1:2);
    dough_d1_seq = dough_d1(dough_start:dough_end,1:2);
    dough_area = dough(dough_start:dough_end,3)
    
    clear norm
    Dough_frames = zeros(4,4,length(dough_seq));
    Dough_areas = zeros(1,length(dough_seq));
    for kk=1:length(dough_seq)
        dough_t = normalizeNadia(dough_seq(kk,:)',fc,cc,kc,alpha_c)*Tc_ext(3)*0.001;
        dough_d1_t = normalizeNadia(dough_d1_seq(kk,:)',fc,cc,kc,alpha_c)*Tc_ext(3)*0.001;

        Dough_D1 = eye(4); Dough_D1(1:3,4) = [dough_d1_t(1) dough_d1_t(2) Tc_ext(3)*0.001]';
        Dough_D1 = Camera_Origin_In_World*Dough_D1;

        Dough_Center = eye(4); Dough_Center(1:3,4) = [dough_t(1) dough_t(2) Tc_ext(3)*0.001]';
        Dough_Center = Camera_Origin_In_World*Dough_Center;

        xdir = -(Dough_Center(1:3,4)-Dough_D1(1:3,4))/norm(Dough_Center(1:3,4)-Dough_D1(1:3,4));    
        zdir = -(Dough_Center(1:3,4) - Camera_Origin_In_World(1:3,4))/norm(Dough_Center(1:3,4)- Camera_Origin_In_World(1:3,4));
        ydir = cross(zdir,xdir)/norm(cross(zdir,xdir));
        Dough_Center (1:3,1:3) = [xdir ydir zdir];
        
        Dough_frames(:,:,kk) = Dough_Center;
        
        pixel_area_len = ceil(sqrt(dough_area(kk)));
        tl_corn = normalizeNadia([0 0]',fc,cc,kc,alpha_c)*Tc_ext(3)*0.001;
        a_width = normalizeNadia([0 pixel_area_len]',fc,cc,kc,alpha_c)*Tc_ext(3)*0.001;
        a  = sqrt(sum((tl_corn - a_width) .^ 2));
        area = a^2;
        Dough_areas(kk) = area;
    end
    
    TrajSeg = [];
    TrajSeg = Segm_results_clean{seq(j),1};
    Traj = EE_PnP{seq(j),1};
    SegPoints = [1; TrajSeg(:,2)];
    dough_frame_id = 1;

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
          
            Corr_Dough_Frame = Dough_frames(:,:,dough_frame_id);
            Corr_Dough_Area = Dough_areas(dough_frame_id);
            drawframe(Corr_Dough_Frame,0.03)
            hold on 
            
            plot3(X,Y,Z,'Color', color(TrajSeg(i,1),:),'LineWidth', 3);
            hold on
            text(Traj(1,SegPoints(i)),Traj(2,SegPoints(i)),Traj(3,SegPoints(i))+0.05,num2str(TrajSeg(i,1)),'FontSize',16, 'Color', color(TrajSeg(i,1),:));
            if TrajSeg(i,1)==1
                drawframe(EE_HnP{seq(j),1}(:,:,SegPoints(i)),0.03)
                drawframe(EE_HnP{seq(j),1}(:,:,SegPoints(i+1)),0.03)
            end
            % Make Data Structure
            clear action
            if (TrajSeg(i,1)==3)
                action.description = 'reach';    
            end
            if (TrajSeg(i,1)==1)
                action.description = 'roll';    
            end
            if (TrajSeg(i,1)==2)
                action.description = 'back';    
            end
            action.BP_HMM_id = TrajSeg(i,1);
            action.sampling_ratio = 5;
            action.dough_area = Corr_Dough_Area; 
            action.dough_frame_in_world = Corr_Dough_Frame;

            action.in_world.EE_FOR = EE_FTnP{seq(j),1}(1:3,start_seg:end_seg);
            action.in_world.EE_TQS = EE_FTnP{seq(j),1}(4:6,start_seg:end_seg);
            action.in_world.EE_Hfull =  EE_HnP{seq(j),1}(:,:,start_seg:end_seg);
            
            EE_Hfull_in_D = zeros(size(action.in_world.EE_Hfull));
            FOR_in_D = zeros(size(action.in_world.EE_FOR));
            TQS_in_D = zeros(size(action.in_world.EE_TQS));
            
            for kk=1:size(action.in_world.EE_Hfull,3)
                EE_in_W = action.in_world.EE_Hfull(:,:,kk);
                EE_in_D = (action.dough_frame_in_world^-1)*EE_in_W;
                EE_Hfull_in_D(:,:,kk) = EE_in_D;
                
                FOR_in_W = action.in_world.EE_FOR(:,kk);
                F = (EE_in_D^-1)*[FOR_in_W; 1];
                FOR_in_D(:,kk) = F(1:3);
                
                TQS_in_W = action.in_world.EE_TQS(:,kk);
                tau = (EE_in_D^-1)*[TQS_in_W; 1];
                TQS_in_D(:,kk) = tau(1:3);
            end
            
            H = EE_Hfull_in_D;
            R = H(1:3,1:3,:);
            t = reshape(H(1:3,4,:),3,length(H));
            q = quaternion(R,true);
              
            action.in_dough.EE_POS = t;
            action.in_dough.EE_ORI = q;
            action.in_dough.EE_FOR = FOR_in_D;
            action.in_dough.EE_TQS = TQS_in_D;
            action.in_dough.EE_Hfull =  EE_Hfull_in_D;
            
            action.in_dough_interp.EE_POS = [];
            action.in_dough_interp.EE_ORI = [];
            action.in_dough_interp.EE_FOR = [];
            action.in_dough_interp.EE_TQS = [];
            
            if (action.BP_HMM_id == 3)
                Phase1 = [Phase1 action];
            end
            if (action.BP_HMM_id == 1)
                Phase2 = [Phase2 action];
            end
            if (action.BP_HMM_id == 2)
                Phase3 = [Phase3 action];
            end    
                        
            if (TrajSeg(i,1)==2)
                dough_frame_id = dough_frame_id + 1;
            end   
    end
   
    
    scatter3(Traj(1,1),Traj(2,1),Traj(3,1), 100, [0 1 0], 'filled')
    scatter3(Traj(1,end),Traj(2,end),Traj(3,end),100, [1 0 0], 'filled')
    scatter3(Camera_Origin_In_World(1,4),Camera_Origin_In_World(2,4),Camera_Origin_In_World(3,4),50, [0 0 0], 'filled')
    
    drawframe(Table_Origin,0.07)
    drawframe(Table_Edge1,0.03)
    drawframe(Table_Edge2,0.03)
    drawframe(Table_Edge3,0.03)
    drawframe(Table_Edge4,0.03)
    drawframe(Chess_Origin,0.03)
    drawframe(Camera_Origin_In_World,0.1)
    fill3([Table_Edge1(1,4) Table_Edge2(1,4) Table_Edge3(1,4) Table_Edge4(1,4)],[Table_Edge1(2,4) Table_Edge2(2,4) Table_Edge3(2,4) Table_Edge4(2,4)],[Table_Edge1(3,4) Table_Edge2(3,4) Table_Edge3(3,4) Table_Edge4(3,4)],[0.5 0.5 0.5])
    
    drawframe(Origin,0.1)
        
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
plot(Phase2(i).in_world.EE_FOR'); legend('fx','fy','fz')
subplot(2,1,2)
plot(Phase2(i).in_world.EE_TQS'); legend('taux','tauy','tauz')

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
plot(Phase2(i).in_dough.EE_FOR'); legend('fx','fy','fz')
subplot(2,1,2)
plot(Phase2(i).in_dough.EE_TQS'); legend('taux','tauy','tauz')


%% Interpolate data
for jj=1:1
    figure('Color',[1 1 1])
    % for ii=1:length(Phase1)
    for ii=1:5
        subplot(5,1,ii)
        pos = Phase1(jj*ii).in_dough.EE_POS;
        plot(pos')
    end
end

%% Interpolation station
%---- PHASE 1 ---%
min_lengths = length(Phase1(1).in_dough.EE_POS);
for ii=1:length(Phase1)   
        min_lengths = [min_lengths length(Phase1(ii).in_dough.EE_POS)];
end 
mean_length = ceil(mean(min_lengths));
min_length = min(min_lengths);
k = 1;
for jj=1:9
    figure('Color',[1 1 1])
    for ii=1:5
        subplot(5,1,ii)
        Phase = Phase1(k); 
        POS = Phase.in_dough.EE_POS;
        New_POS = interpolateSpline(POS,min_length);

        ORI = Phase.in_dough.EE_ORI;
        New_ORI = interpolateSpline(ORI,min_length);
        
        FOR = Phase.in_dough.EE_FOR;
        New_FOR = interpolateSpline(FOR,min_length);
        
        TQS = Phase.in_dough.EE_TQS;
        New_TQS = interpolateSpline(TQS,min_length);
        
        Interp_Data = [New_POS; New_ORI; New_FOR; New_TQS];
        plot(Interp_Data')
        name=strcat('Sequence',num2str(k))
        title(name)
        Phase1(k).in_dough_interp.EE_POS = New_POS;
        Phase1(k).in_dough_interp.EE_ORI = New_ORI;
        Phase1(k).in_dough_interp.EE_FOR = New_FOR;
        Phase1(k).in_dough_interp.EE_TQS = New_TQS;
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
min_length = min(min_lengths);
k = 1;
for jj=1:9
    figure('Color',[1 1 1])
    for ii=1:5
        subplot(5,1,ii)
        Phase = Phase2(k); 
        POS = Phase.in_dough.EE_POS;
        New_POS = interpolateSpline(POS,min_length);

        ORI = Phase.in_dough.EE_ORI;
        New_ORI = interpolateSpline(ORI,min_length);
        
        FOR = Phase.in_dough.EE_FOR;
        New_FOR = interpolateSpline(FOR,min_length);
        
        TQS = Phase.in_dough.EE_TQS;
        New_TQS = interpolateSpline(TQS,min_length);
        
        Interp_Data = [New_POS; New_ORI; New_FOR; New_TQS];
        plot(Interp_Data')
        name=strcat('Sequence',num2str(k))
        title(name)
        Phase2(k).in_dough_interp.EE_POS = New_POS;
        Phase2(k).in_dough_interp.EE_ORI = New_ORI;
        Phase2(k).in_dough_interp.EE_FOR = New_FOR;
        Phase2(k).in_dough_interp.EE_TQS = New_TQS;
        k=k+1;
    end
end

%% Interpolation station
%---- PHASE 3 ---%
min_lengths = length(Phase3(1).in_dough.EE_POS);
for ii=1:length(Phase3)   
        min_lengths = [min_lengths length(Phase3(ii).in_dough.EE_POS)];
end 
mean_length = ceil(mean(min_lengths));
min_length = min(min_lengths);
k = 1;
for jj=1:9
    figure('Color',[1 1 1])
    for ii=1:5
        subplot(5,1,ii)
        Phase = Phase3(k); 
        POS = Phase.in_dough.EE_POS;
        New_POS = interpolateSpline(POS,min_length);

        ORI = Phase.in_dough.EE_ORI;
        New_ORI = interpolateSpline(ORI,min_length);
        
        FOR = Phase.in_dough.EE_FOR;
        New_FOR = interpolateSpline(FOR,min_length);
        
        TQS = Phase.in_dough.EE_TQS;
        New_TQS = interpolateSpline(TQS,min_length);
        
        Interp_Data = [New_POS; New_ORI; New_FOR; New_TQS];
        plot(Interp_Data')
        name=strcat('Sequence',num2str(k))
        title(name)
        Phase3(k).in_dough_interp.EE_POS = New_POS;
        Phase3(k).in_dough_interp.EE_ORI = New_ORI;
        Phase3(k).in_dough_interp.EE_FOR = New_FOR;
        Phase3(k).in_dough_interp.EE_TQS = New_TQS;
        k=k+1;
    end
end

%% Save Segmentation Data
save('Series3_Segmented.mat','Phase1','Phase2','Phase3')


%% Plot Clean Segmentation Results

%plotSegDataNadiaResults(data, [4 5 6], Segm_results_clean)

reach = 3; roll=1; back=2;

primitive = back;

Mu = bestGauPsi.theta(primitive).mu;
Precision = bestGauPsi.theta(primitive).invSigma; 
Sigma = bestGauPsi.theta(primitive).invSigma^-1; 
Correlation = cov2cor(Sigma); %Correlation
tmp_corr = Correlation;
pearson_factor = 0.7; %Pearson's Correlation Factor
Trunc_Correlation = tmp_corr; Trunc_Correlation(Trunc_Correlation>-pearson_factor&Trunc_Correlation<pearson_factor) = 0;

figure('Color',[1 1 1])
subplot(3,2,1)
imagesc(Precision)
title('Precision')
colorbar

subplot(3,2,2)
imagesc(Sigma)
title('Covariance')
colorbar

subplot(3,2,3)
imagesc(Correlation)
title('Correlation (Normalized Covariance)')
colorbar

subplot(3,2,4)
imagesc(Trunc_Correlation)
title('Truncated Correlation')
colorbar

% Gaussian Distribution Sample

% scale by the square root (see http://en.wikipedia.org/wiki/Cholesky_decomposition) of sigma

for i=1:12
R = randn(12, 1);
R1 = chol(Sigma)*R;
subplot(3,2,5)
plot(R1, 'LineStyle','--','Marker','+','Color', [rand rand rand])
hold on
end
xlabel('variable index')
ylabel('covariance sample')
title('Sample from Covariance')

for i=1:12
R = randn(12, 1);
R2 = chol(Correlation)*R;
subplot(3,2,6)
plot(R2, 'LineStyle','--','Marker','+','Color', [rand rand rand])
hold on
end
xlabel('variable index')
ylabel('covariance sample')
title('Sample from Norm. Covariance (Corr)')