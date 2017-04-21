%% Lucia's Carrot Grating Experiments
clear all
clc
d = '/home/nadiafigueroa/dev/MATLAB/michaelchughes-NPBayesHMM/code/demo/PancakeFlipping/';

% KUKA Raw Data
subdir = strcat(d,'KUKArawdata_flipping/');

% PR2 Raw Data
% subdir = strcat(d,'PR2rawdata_new/');

files = dir(strcat(subdir,'*.txt'));
data = {};
for ii=1:length(files)
% for ii=1:1    
    filename = strcat(subdir,files(ii,1).name);    
    data{ii,1} = textread(filename);
end

%% KUKA Extract Poses/Forces
Hn = {};
Fn = {};
for jj=1:length(data)
    poses = [];
    Hpose = data{jj};
    H1i = [Hpose(:,8:11)'];  
    H1ii = reshape(H1i,1,4,length(H1i));
    H2i = [Hpose(:,12:15)'];  
    H2ii = reshape(H2i,1,4,length(H2i));
    H12ii = cat(1,H1ii,H2ii);
    H3i = [Hpose(:,16:19)'];  
    H3ii = reshape(H3i,1,4,length(H3i));
    H13 = cat(1,H12ii,H3ii);
    norm = [0 0 0 1];
    n = repmat(norm',1,length(H13));
    N = reshape(n,1,4,length(H13));
    H = cat(1,H13,N);
    Hn{jj,1} = H;
    Fn{jj,1} = [Hpose(:,34:39)'];      
end

clear data H H12ii H13 H1i H1ii H2i H2ii H3i H3ii Hpose n N

%% PR2 Extract Poses
Hn = {};
Xn = {};
for jj=1:length(data)
    Xn{jj,1} = data{jj,1}';
    Hn{jj,1} = fromXtoH(Xn{jj,1},1);    
end

%% Visualize position trajectories
figure('Name', 'PR2 Flipping Recording','Color',[1 1 1])
% for j=1:3:length(Hn)
for j=1:3:28
% for j=9:12
    h = Hn{j,1};    
    plot3(reshape(h(1,4,:),1,length(h)),reshape(h(2,4,:),1,length(h)),reshape(h(3,4,:),1,length(h)),'Color',[rand rand rand],'LineWidth', 2);
    hold on 
end
xlabel('x');ylabel('y');zlabel('z');
grid on
axis equal
%% Visualize Rigid Body Trajectories
t= 1;
named_figure('Pancake Flip Trial Raw ', t);
clf;
drawframetraj(Hn{t},0.01,1);
%%
t= 3;
named_figure('Pancake Flip Trial Raw ', t);
clf;
drawframetraj(Hn{t},0.01,1);

%% Convert to 7D for Segmentation fro KUKA measurements
% Xn = {};
XnP = {};
FnP = {};
HnP = {};
for ii=1:length(Hn)
%     Original Data
     H = Hn{ii,1};
    % Transformed Data
%     H = HnT{ii,1};
    R = H(1:3,1:3,:);
    t = reshape(H(1:3,4,:),3,length(H));
    q = quaternion(R);
    X = cat(1,t,q);
%     Xn{ii,1} = X;
    XnP{ii,1} = preprocessMocapData(X,7,2);
    FnP{ii,1} = preprocessMocapData(Fn{ii,1},7,2);
    HnP{ii,1} = fromXtoH(XnP{ii,1},1);
end

clear X Fn Hn

%% Convert to 7D for Segmentation for PR2 H trajectories
XnT = [];
for ii=1:length(Hn)
    H = Hn{ii,1};
    R = H(1:3,1:3,:);
    t = reshape(H(1:3,4,:),3,length(H));
    q = quaternion(R);
    X = cat(1,t,q);
    XnT{ii,1} = X;
end



%% Visualize subsampled position trajectories
figure('Color',[1 1 1])
for j=1:3:length(HnP)
% for j=9:12
    h = HnP{j,1};    
    plot3(reshape(h(1,4,:),1,length(h)),reshape(h(2,4,:),1,length(h)),reshape(h(3,4,:),1,length(h)),'Color',[rand rand rand],'LineWidth', 2);
    hold on 
end
grid on

%% Visualize Rigid Body Trajectories
t= 1;
named_figure('PancakeFlip Pre-processed', t);
clf;
drawframetraj(HnP{t},0.01,1);
%%
t= 12;
named_figure('PancakeFlip Pre-processed', t);
clf;
drawframetraj(HnP{t},0.01,1);

%% View Measurements
figure('Color',[1 1 1]);
Tot = length(Xn);
Tot = 
int = 1;
rows = floor(Tot/int);
for kk=1:int:Tot
    if kk==1
        row_int =1
    else
        row_int = floor(kk/int)
    end
subplot(rows,1,row_int)
plot(Xn{kk}')
% legend('x','y','z','qw','qi','qj','qz')
end

%% Mix PR2 and KUKA measurements
XnMix = {};
XnMix = {XnP{1:8,1} XnT{1:8,1}}

%% Preprocess for segmentation (From MoCap data pre-processing experiment)

RobotChannelNames = {'pos.x','pos,y','pos.z', 'pos.qw','pos.qi','pos.qj','pos.qz'};

N = length(XnP);
% N = 18;
% N = length(XnMix);
% nDim = 7;
nDim = 3;
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
    
    % Poses
%     D = XnT{ii};
    D = XnP{ii}(1:3,:);
    % Forces
%     D = Fn{ii,1};
    %KUKA Poses + Forces
%      D = cat(1,XnP{ii},FnP{ii,1}/20);    
    %KUKA + PR2 Poses
%      D = XnMix{ii};


%     % Enforce zero-mean
    D = D';
    D = bsxfun( @minus, D, mean(D,2) );
    D = D';
   
    %Pre-process Mocap Data
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

%% Plot Segmented Trajectory
seq = [1 2];
figure('Color',[1 1 1])
fig_row = floor(sqrt(length(seq)));
fig_col = length(seq)/fig_row;
for i=1:max(Total_feats)
    c(i,:) = [rand rand rand];
end
for j=1:length(seq)
    subplot(fig_row,fig_col,j)
    TrajSeg = [];
    TrajSeg = Segm_results{seq(j),1};
    Traj = XnP{seq(j),1};
%     Traj = XnMix{seq(j)};
    SegPoints = [1; TrajSeg(:,2)]
    for i=1:length(TrajSeg)
        plot3(Traj(1,SegPoints(i):SegPoints(i+1)),Traj(2,SegPoints(i):SegPoints(i+1)),Traj(3,SegPoints(i):SegPoints(i+1)),'Color', c(TrajSeg(i,1),:),'LineWidth', 3);
        hold on
    end
    xlabel('x');ylabel('y');zlabel('z');
    axis equal
    grid on
end
