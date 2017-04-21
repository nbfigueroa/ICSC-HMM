%% Convert 7d (x,y,z,qw,qi,qj,qk) Trajectories to Homogeneous Matrices (Ridid Body Motion Trajectories)
%rescale quaternion
Xnr = Xn;
for j=1:length(Xn)
Xnr{j}(4:7,:) = Xn{j}(4:7,:)/50; 
data = Xnr{j};
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
drawframetraj(Hn{5},1,1);
% animframetraj(Hn{1});
%  nice3d()
%  hold on;

%% Convert Homogenous Rigid Motion Representation (H=4x4) to Twist/Screw parameters full trajectory (w1,v1,w2,v2,w3,v3)
for j=1:length(Hn)   
% for j=1:3
H = Hn{j,1};
truelabels = TL{j,1};
Tints = Tn{j,1};
fprintf('----------------------------\n');
fprintf('Sequence %d Length: %d\n',j, length(H));

    Xm = Xnr{j,1};
%     Mtype = unique(truelabels);
    Mtype = j;
    x = 1:1:length(Xm);
    figure;
    subplot(4,1,1);
    tit1 = strcat(num2str(Mtype), ' xyzq');
    plot(x,Xm); title(tit1)
    legend('x','y','z','qw','qi','qj','qk')
    
    %Compute twist for trajectory j
    tw = zeros(6,size(H,3));
    Hrel = zeros(4,4,size(H,3));
    for i=1:length(H)
        if i==1
            T = H(:,:,i);
            Hrel(:,:,i) = T;
            [xi theta] = homtotwist(T);
            tw(:,i) = [xi*theta];
        else
            T1 = H(:,:,i-1);
            T2 = H(:,:,i);
            T12 = inv(T1)*T2;
            Hrel(:,:,i) = T12;
            [xi theta] = homtotwist(T12);
            tw(:,i) = [xi*theta];
        end
    end
    tw(:,1) = zeros(6,1);
    RelTwist{j,1} = tw;
    RelH{j,1} = Hrel;
    subplot(4,1,2);
    tit2 = strcat(num2str(Mtype),' twist');
    plot(x,tw); title(tit2)
    legend('vi','vj','vk','wi','wj','wk')
        
    %Check relative twist computation (reconstructing original trajectory)
    rel_tw = RelTwist{j,1};
    Hback = zeros(4,4,length(rel_tw));
    Xtwist_rec = zeros(7,length(tw));
    for i=1:length(Hback)
        if i==1
            Hback(:,:,1) = twistexp(rel_tw(:,1));
            Hback(:,:,1) = H(:,:,1);
        else
            Hrel1 = Hback(:,:,i-1);
            Hrel12 = twistexp(rel_tw(:,i));
            Hback(:,:,i) = Hrel1*Hrel12;
        end        
        Hi = Hback(:,:,i) ;
        Xtwist_rec(1:3,i) = Hi(1:3,4);
        Xtwist_rec(4:7,i) = quaternion(Hi(1:3,1:3),true);
    end
    
    subplot(4,1,3);
    tit3 = strcat(num2str(Mtype),' xyzqRec');
    plot(x,Xtwist_rec); title(tit3)
    

    %Computing scew coordinates from twist
    scew = zeros(
    lambda = 2; 
    for i=1:length(tw)
        
    end

end




%% Compute Fernet-Serrat Frames of trajectory and space-curve parameters
for j=1:length(Hn)   
% for j=4:4
H = Hn{j,1};
truelabels = TL{j,1};
Tints = Tn{j,1};
fprintf('----------------------------\n');
fprintf('Sequence %d Length: %d\n',j, length(H));

    Xm = Xnr{j,1};
    Mtype = j;
    x = 1:1:length(Xm);
    figure;
    subplot(4,1,1);
    tit1 = strcat(num2str(Mtype), ' xyzq');
    plot(x,Xm); title(tit1)
    legend('x','y','z','qw','qi','qj','qk')
    
    %Compute twist for trajectory j
    tw = zeros(6,size(H,3));
    Hrel = zeros(4,4,size(H,3));
    for i=1:length(H)
        if i==1
            T = H(:,:,i);
            Hrel(:,:,i) = T;
            [xi theta] = homtotwist(T);
            tw(:,i) = [xi*theta];
        else
            T1 = H(:,:,i-1);
            T2 = H(:,:,i);
            T12 = inv(T1)*T2;
            Hrel(:,:,i) = T12;
            [xi theta] = homtotwist(T12);
            tw(:,i) = [xi*theta];
        end
    end
    

    clear norm
    %Frenet-Serrat Frame Computation
    Xm3d = Xm(1:3,:);
    Xm3d_dot = [zeros(3,1) diff(Xm(1:3,:)')'];
    Xm3d_dotdot = [zeros(3,1) diff(Xm3d_dot')'];
    Xm3d_dotdotdot = [zeros(3,1) diff(Xm3d_dotdot')'];
    fs = zeros(4,4,length(Xm));
    fs(1:4,1:4,1) = eye(4);
    L = zeros(1,length(Xm));
    K =zeros(1,length(Xm));
    Tau = zeros(1,length(Xm));
    for i=2:length(Xm)
        v = tw(1:3,i);
        omega = tw(4:6,i);
        k = norm(cross(Xm3d_dot(:,i),Xm3d_dotdot(:,i)))/(norm(Xm3d_dot(:,i))^3);
        arclength = norm(Xm3d_dot(:,i))*i - norm(Xm3d_dot(:,1));%%kinda sure
        T = Xm3d_dot(:,i)/norm(Xm3d_dot(:,i)); %T=r'/|r'|
        if k==0 %straight line
            tau = 0; %no curvature
            R = quaternion(Xm(4:7,i),true);
            Ry = R(:,2);
            Rz = R(:,3);
            Ry2z = cross(Ry,Rz);
            Rz2T = cross(Rz,T); %Rznew=cross(Rz2T,T)
            N = cross(Ry2z,Rz2T);
            B = cross(N,T);
        else
            tau = dot(cross(Xm3d_dot(:,i),Xm3d_dotdot(:,i)),(Xm3d_dotdotdot(:,i))/(norm(cross(Xm3d_dot(:,i),Xm3d_dotdot(:,i)))^2));
            N = cross(Xm3d_dot(:,i),cross(Xm3d_dotdot(:,i),Xm3d_dot(:,i)))/(norm(Xm3d_dot(:,i))*norm(cross(Xm3d_dotdot(:,i),Xm3d_dot(:,i)))); %T'/|T'|=r'x(r''x r')/|r'||r''x r'|
            B = cross(T,N);
        end
            display('here')
        L(1,i) = arclength;
        K(1,i) = k;
        Tau(1,i) = tau;
        
        fs(1:4,1:4,i) = eye(4);
        fs(1:3,1:3,i) = [T N B];
        fs(1:3,4,i) = Xm3d(:,i);
    end
        display('here')
    FS_frames{j,1} = fs;
   
    subplot(4,1,2);
    tit = strcat(num2str(Mtype),' arclength');
    plot(x,L); title(tit)
    subplot(4,1,3);
    tit = strcat(num2str(Mtype),' curvature');
    plot(x,K); title(tit)
    subplot(4,1,4);
    tit = strcat(num2str(Mtype),' torsion');
    plot(x,Tau); title(tit)
end
%% Visualize Ferret-Serrat Frames
named_figure('Trajectory FS frames 5');
clf;
drawframetraj(FS_frames{5},1,1);

%% Screw axis orientation
scalarAxis = zeros(1,size(tw,2));
for t=2:size(tw,2)
    scalarAxis(1,t) = dot(tw(:,t-1),tw(:,t)); 
end
figure;
plot(x,scalarAxis)
%% Convert Homogenous Rigid Motion Representation (H=4x4) to Twist (xi*theta=6)
for j=1:length(Hn)
H = Hn{j,1};
twTh = zeros(6,size(H,3));
for i=1:length(H)
    T = H(:,:,i);
    [xi theta] = homtotwist(T);
    twTh(:,i) = [xi*theta];
end
TwistTh{j,1} = twTh;
end

%% Test conversion to twist and back
twth = TwistTh{3,1};
Hback = zeros(4,4,length(twth));
Hback(:,:,1) = eye(4);
for i=2:length(Hback)
%     Hback(:,:,i) = twistexp(twth(1:6,i),twth(7,i));
    Hback(:,:,i) = twistexp(twth(1:6,i));
end

named_figure('Trajectory',1);
clf;
drawframetraj(Hback,2,1);


%% Convert Homogenous Rigid Motion Representation (H=4x4) to Relative Twist (xi,theta=7)
for j=1:length(Hn)
H = Hn{j,1};
twTh = zeros(6,size(H,3));
Hrel = zeros(4,4,size(H,3));
for i=1:length(H)
    if i==1
        T = H(:,:,i);
        Hrel(:,:,i) = T;
        [xi theta] = homtotwist(T);
        twTh(:,i) = [xi*theta];
    else
        T1 = H(:,:,i-1);
        T2 = H(:,:,i);
        T12 = inv(T1)*T2;
        Hrel(:,:,i) = T12;
        [xi theta] = homtotwist(T12);
        twTh(:,i) = [xi*theta];
    end
end
RelTwistTh{j,1} = twTh;
RelH{j,1} = Hrel;
end
%% Test relative transformations
Hrel = RelH{1,1};
Hback_rel = zeros(4,4,size(Hrel,3));
for j=1:length(Hrel)
    if j==1
        Hback_rel(:,:,j) = Hrel(:,:,1);
    else
        Hrel1 = Hback_rel(:,:,j-1);
        Hrel12 = Hrel(:,:,j);
        Hback_rel(:,:,j) = Hrel1*Hrel12;
    end
end
named_figure('Trajectory',2);
clf;
drawframetraj(Hback_rel,2,1);
%% Test conversion to relative twist and back
rel_tw = RelTwistTh{1,1};
Hback = zeros(4,4,length(rel_tw));
Hback(:,:,1) = twistexp(rel_tw(:,1));
for i=2:length(Hback)
%     Hback(:,:,i) = twistexp(twth(1:6,i),twth(7,i));
    Hrel1 = Hback(:,:,i-1);
    Hrel12 = twistexp(rel_tw(:,i));
    Hback(:,:,i) = Hrel1*Hrel12;
end

named_figure('Trajectory',1);
clf;
drawframetraj(Hback,2,1);

%% Preprocess for segmentation (From MoCap data pre-processing experiment)

% RobotChannelNames = {'EndEffector.x', 'EndEffector.y', 'EndEffector.z', ... 
%     'EndEffector.qw', 'EndEffector.qx', 'EndEffector.qy','EndEffector.qz'};
RobotChannelNames = {'EndEffector.v1','EndEffector.v2','EndEffector.v3', ...
    'EndEffector.w1', 'EndEffector.w3', 'EndEffector.w3'};


% nDim = 8;
nDim = 6;
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
for ii = 1:length(RelTwist)
%     
%     D = Xn{ii};   
%     Dn = XnNoise{ii};
    
    D = RelTwist{ii}(:,:);   
%     D = D';    
%     D = bsxfun( @minus, D, mean(D,2) );
%     D = D';
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
