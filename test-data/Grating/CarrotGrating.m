%% Change Reference Frame for Demonstrationsc
data_path = './test-data/'
            load(strcat(data_path,'/Grating/CarrotGrating_robot.mat'))            
%%
offset = [0.65 -0.15 -0.05 0 0 0 0]';
R = rotz(-pi/4);

for d=1:length(Data)
    data = Data{d}' + repmat(offset,[1 length(Data{d})]);
    for dd=1:length(data)
        data(1:3,dd) = R*data(1:3,dd);
    end
    Data{d} = data';
end

%% Lucia's Carrot Grating Experiments
clear all
clc
d = './';

% Raw Data
subdir = strcat(d,'rawdata/');
files = dir(strcat(subdir,'*.txt'));
Hn = {};
for ii=1:length(files)
% for ii=1:1    
    filename = strcat(subdir,files(ii,1).name);
    poses = [];
    Hpose = textread(filename);
    H1i = [Hpose(:,1)'; Hpose(:,2)'; Hpose(:,3)'; Hpose(:,4)'];  
    H1ii = reshape(H1i,1,4,length(H1i));
    H2i = [Hpose(:,5)'; Hpose(:,6)'; Hpose(:,7)'; Hpose(:,8)'];  
    H2ii = reshape(H2i,1,4,length(H2i));
    H12ii = cat(1,H1ii,H2ii);
    H3i = [Hpose(:,9)'; Hpose(:,10)'; Hpose(:,11)'; Hpose(:,12)'];  
    H3ii = reshape(H3i,1,4,length(H3i));
    H13 = cat(1,H12ii,H3ii);
    norm = [0 0 0 1];
    n = repmat(norm',1,length(H13));
    N = reshape(n,1,4,length(H13));
    H = cat(1,H13,N);
    Hn{ii,1} = H; 
end

%% PLot Values
figure('Color', [1 1 1]); 
plot(Xn{1}')
xlabel('time')
legend('x','y','z','q_w','q_i','q_j','q_k')

%% Convert to 7D for Segmentation
Xn = [];
figure
for ii=1:length(Hn)
%     Original Data
     H = Hn{ii,1};
    % Transformed Data
%     H = HnT{ii,1};
    R = H(1:3,1:3,:);
    t = reshape(H(1:3,4,:),3,length(H));
    q = quaternion(R);
    X = cat(1,t,q);
    Xn{ii,1} = X;
    subplot(length(Hn),1,ii)
    plot(X')
end

%% Visualize all trajectories
figure('Color',[1 1 1])
for j=1:3:length(Hn)
% for j=9:12
    h = Hn{j,1};    
    plot3(reshape(h(1,4,:),1,length(h)),reshape(h(2,4,:),1,length(h)),reshape(h(3,4,:),1,length(h)),'Color',[rand rand rand]);
    hold on 
    grid on
end

%% Tranform Original Trajectories to Different Reference Frames
W1 = eye(4);
W2 = W1;
W2(1:3,1:3) = rotz(-pi/2);
W2(1:3,4) = [-1 0.5 0];

W3 = W1;
W3(1:3,1:3) = rotz(pi/5)*roty(pi/2);
W3(1:3,4) = [-0.5 0.1 1];

HnT = Hn;
HnT(9:12,:)=[];
for ii=1:3
    h = Hn{ii+4,1}; 
    for jj=1:length(h)
        Hjj = h(:,:,jj);
        Htemp = W3*Hjj;
        Htemp(1:3,4) = Htemp(1:3,4)*2;
        h(:,:,jj) = Htemp;
    end
    HnT{ii+3,1} = h;
end

for ii=1:3
    h = Hn{ii+8,1}; 
    for jj=1:length(h)
        Hjj = h(:,:,jj);
        Htemp = W2*Hjj;
        Htemp(1:3,4) = Htemp(1:3,4)*0.7;
        h(:,:,jj) = Htemp;
    end
    HnT{ii+6,1} = h;
end

figure('Color',[1 1 1])
for j=1:length(HnT)
    h = HnT{j,1};    
    plot3(reshape(h(1,4,:),1,length(h)),reshape(h(2,4,:),1,length(h)),reshape(h(3,4,:),1,length(h)),'Color',[rand rand rand]);
    hold on 
end

drawframe(W1,0.1)
drawframe(W2,0.1)
drawframe(W3,0.1)
grid on


%% Visualize Rigid Body Trajectories
t= 1;
named_figure('Carrot Grating Trial Raw ', t);
clf;
drawframetraj(Hn{t},0.01,1);

t= 12;
named_figure('Carrot Grating Trial Raw ', t);
clf;
drawframetraj(Hn{t},0.01,1);



%% Add Null behaviours
% Xn = [];
XnNull = [];
for jj=1:length(Xn)
% for jj=1:1
   Xnew = [];
   Xo = Xn{jj,1};
   [num id]=max(abs(rand(3,1)));
   null_start = floor(length(Xo)*id/3);
   null_length = max(length(Xo)/2,floor(rand(1)*1000));
   Xnew = [Xo(:,1:null_start-1) repmat(Xo(:,null_start),1,null_length) Xo(:,null_start:end)];
   figure
   subplot(2,1,1)
   plot(Xo');
   subplot(2,1,2)
   plot(Xnew')
   XnNull{jj,1} = Xnew;
end

%% Aligned Data
clear all
clc
d = '/home/nadiafigueroa/dev/MATLAB/michaelchughes-NPBayesHMM/code/demo/CarrotGratting/';
subdir = strcat(d,'aligneddata/');
files = dir(strcat(subdir,'*.txt'));
for ii=1:length(files)
    pos = [];
    filename = strcat(subdir,files(ii,1).name);
    pos = textread(filename);   
    Xn_sep{ii,1} = pos;
end

%Merge measurements
for jj=1:size(pos,1)
    X = [];
    X(1,:) = Xn_sep{1}(jj,:);
    X(2,:) = Xn_sep{2}(jj,:);
    X(3,:) = Xn_sep{3}(jj,:);
    Xn{jj,1} = X;
end

%% Add frame to measurements
for j=1:length(Xn)
% for j=1:1
data = Xn{j,1};
R = repmat(eye(3),1,length(data));
R3 = reshape(R,3,3,length(data));
t = reshape(data(1:3,:),3,1,size(data,2));
norm = [0 0 0 1];
n = repmat(norm',1,size(data,2));
N = reshape(n,1,4,size(data,2));
H_data = cat(2,R3,t);
H = cat(1,H_data,N);
Hn{j,1} = H;
end

%% Visualize Rigid Body Trajectories
t=9;
named_figure('Carrot Grating Trial ', t);
clf;
drawframetraj(Hn{t},0.01,1);


% Convert to 7D for Segmentation
XnT = [];
figure
for ii=1:length(HnT)
%     Original Data
%     H = Hn{ii,1};
    % Transformed Data
    H = HnT{ii,1};
    R = H(1:3,1:3,:);
    t = reshape(H(1:3,4,:),3,length(H));
    q = quaternion(R);
    X = cat(1,t,q);
    XnT{ii,1} = X;
   subplot(length(HnT),1,ii)
   plot(X');
end


%% Preprocess for segmentation (From MoCap data pre-processing experiment)
clear all
RobotChannelNames = {'pos.x','pos,y','pos.z', 'pos.qw','pos.qi','pos.qj','pos.qz'};
load('transformed_grating_example.mat')

N = length(XnT);
nDim = 7;
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
    
    % All values
    D = XnT{ii}(:,:);  
           
    scatter3(D(1,:),D(2,:),D(3,:),5, [rand rand rand],'filled')
    hold on
    
%     % Enforce zero-mean
    D = D';
    D = bsxfun( @minus, D, mean(D,1) );
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
% %         end
%     end
%     Truelabels = ones(1,length(D))*ii; 
%     TL{ii,1} = zeros(1,length(D)); 
    % Add the sequence to our data structure
    Robotdata = Robotdata.addSeq( D, num2str(ii));
    data_struct(ii).obs = D;
%     data_struct(ii).true_labels = Truelabels;
end
grid on
axis ij
xlabel('x')
ylabel('y')
zlabel('z')
% -------------------------------------- Preproc autoregressive data
% This step simply builds necessary data structs for efficient AR inference
%   including filling in XprevR data field so that for any time tt,
%       XprevR(:,tt) = [Xdata(:, tt-1); Xdata(:,tt-2); ... Xdata(:,tt-R) ]
%   allowing of course for discontinuities at sequence transitions
data = Robotdata;
RobotdataAR = ARSeqData( Preproc.R, Robotdata);

%% Plot Segmented Trajectory
seq = [1 4 8];
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
    Traj = Xn{seq(j),1};
    SegPoints = [1; TrajSeg(:,2)]
    for i=1:length(TrajSeg)
        plot3(Traj(1,SegPoints(i):SegPoints(i+1)),Traj(2,SegPoints(i):SegPoints(i+1)),Traj(3,SegPoints(i):SegPoints(i+1)),'Color', c(TrajSeg(i,1),:),'LineWidth', 3);
        hold on
    end
    xlabel('x');ylabel('y');zlabel('z');
    axis equal
    grid on
end