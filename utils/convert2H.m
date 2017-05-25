%% Convert 7d (x,y,z,qw,qi,qj,qk) Trajectories to Homogeneous Matrices (Ridid Body Motion Trajectories)
for j=1:length(Xn)
data = Xn{j};
R = quaternion([data(4,:);data(5,:);data(6,:);data(7,:)], true);
t = reshape(data(1:3,:),3,1,size(data,2));
norm = [0 0 0 1];
n = repmat(norm',1,size(data,2));
N = reshape(n,1,4,size(data,2));
H_data = cat(2,R,t);
H = cat(1,H_data,N);
Hn{j,1} = H;
end

%% Visualize Rigid Body Trajectories
named_figure('Trajectory',1);
clf;
drawframetraj(Hn{1},2,1);
% animframetraj(Hn{1});
%  nice3d()
%  hold on;

%% Convert Homogenous Rigid Motion Representation (H=4x4) to Time-based Invariants (w1,v1,w2,v2,w3,v3)
for j=1:length(Hn)
H = Hn{j,1};
truelabels = TL{j,1};
tb_inv = zeros(6,size(H,3));
for i=1:length(H)
    T = H(:,:,i);
    [xi theta] = homtotwist(T);
    twist = [xi];
    
end
TB_INV{j,1} = tb_inv;
end

%% Convert Homogenous Rigid Motion Representation (H=4x4) to Twist (xi,theta=7)
for j=1:length(Hn)
H = Hn{j,1};
twTh = zeros(6,size(H,3));
for i=1:length(H)
    T = H(:,:,i);
    [xi theta] = homtotwist(T);
    twTh(:,i) = [xi];
end
TwistTh{j,1} = twTh;
end

%% Preprocess for segmentation (From MoCap data pre-processing experiment)

% RobotChannelNames = {'EndEffector.x', 'EndEffector.y', 'EndEffector.z', ... 
%     'EndEffector.qw', 'EndEffector.qx', 'EndEffector.qy','EndEffector.qz'};
RobotChannelNames = {'EndEffector.v1','EndEffector.v2','EndEffector.v3', ...
    'EndEffector.w1', 'EndEffector.w3', 'EndEffector.w3', 'EndEffector.theta'};


% nDim = 8;
nDim = 7;
Preproc.nObj = N;
Preproc.obsDim = nDim;
Preproc.R = 1;
Preproc.windowSize = nDim;
Preproc.channelNames = RobotChannelNames;

% Create data structure
Robotdata = SeqData();
% RobotdataN = SeqData();

% Create data structure for PR2 Code
% data_struct = struct;

% Read in every sequence, one at a time
% for ii = 1:length( Xn )
for ii = 1:length(TwistTh)
%     
%     D = Xn{ii};   
%     Dn = XnNoise{ii};
    
    D = TwistTh{ii}(:,:);   
    D = D';    
    D = bsxfun( @minus, D, mean(D,2) );
    D = D';
%     
%     Dn = XnNoise{ii}(1:7,:);
%     % Enforce zero-mean
%     Dn = Dn';
%     Dn = bsxfun( @minus, Dn, mean(Dn,2) );
%     Dn = Dn';
    
%     D = preprocessMocapData(D, Preproc.windowSize,2);
    %Scale each component of the observation vector so empirical variance
    %Renormalize so for each feature, the variance of the first diff is 1.0
    %of 1st difference measurements is = 1
%     varFirst =  var(diff(D'));
%     for jj=1:length(varFirst)
%         D(jj,:) = D(jj,:)./sqrt(varFirst(jj));
%     end   
%     
%     varFirstN =  var(diff(Dn'));
%     for jj=1:length(varFirstN)
%         Dn(jj,:) = Dn(jj,:)./sqrt(varFirstN(jj));
%     end   
    

%     
    Tints = Tn{ii,1};
    labels = Ztrue{ii,1};
    for jj=1:length(Tints)
        if jj==1
            Truelabels = repmat(labels(jj),[1 Tints(jj)]); 
        else
            temp = repmat(labels(jj),[1 Tints(jj)]);
            Truelabels = [Truelabels temp];
        end
    end
    % Add the sequence to our data structure
    Robotdata = Robotdata.addSeq( D, num2str(ii), Truelabels);
%     RobotdataN = RobotdataN.addSeq(Dn, num2str(ii), Truelabels);
%     data_struct(ii).obs = D;
%     data_struct(ii).true_labels = Truelabels;
end

% -------------------------------------- Preproc autoregressive data
% This step simply builds necessary data structs for efficient AR inference
%   including filling in XprevR data field so that for any time tt,
%       XprevR(:,tt) = [Xdata(:, tt-1); Xdata(:,tt-2); ... Xdata(:,tt-R) ]
%   allowing of course for discontinuities at sequence transitions

RobotdataAR = Robotdata; % keeping odata around lets debug with before/after
% RobotdataARN = RobotdataN;
RobotdataAR = ARSeqData( Preproc.R, Robotdata);
% RobotdataARN = ARSeqData( Preproc.R, RobotdataN);

%% Test conversion to twist and back
twth = TwistTh{2,1};
Hback = zeros(4,4,length(twth));
Hback(:,:,1) = eye(4);
for i=2:length(Hback)
    Hback(:,:,i) = twistexp(twth(1:6,i),twth(7,i));
%     Hback(:,:,i) = twistexp(twth(1:6,i));
end

named_figure('Trajectory',1);
clf;
drawframetraj(Hback,2,1);

%% Visualizations

named_figure('skew');
for i=1:10, 
  clf;
  drawskewtraj(randskew(),0:pi/20:1/2*pi);
  nice3d; 
end

named_figure('twist');
for i=1:10,
  clf;
  drawtwisttraj(randtwist(),0:pi/20:1/2*pi);
%   nice3d; 
end

named_figure('3 DOF robot');
clf;
r = robot({randtwist('r');randtwist('t');randtwist('s')},randhom());
fk = fkine(r, [0:pi/50:pi ; 0:pi/25:2*pi ; 0:pi/25:2*pi]);
animframetraj(fk);
