%% Bidan's Bottle Cap Opening Experiments
% fringerforce: 1D. Normal force applied on the object surface.
% forcetorque: 6D (fx, fy, fz, tx, ty, tz). Force and torque measured from the bottle cap. tz is the driving torque.
% jointAngles: 23D. Finger joint angles.
% trackWrist: 3D (x, y, z). Position of the wrist.
% trackCap: 3D(x,y,z). Position of the bottle cap.
% angle: 3D(x,y,z). Rotation angle of the cap. The cap rotates around the z axis.
% time: 1D. 

% Load data
clear all
clc
%% raw data
dir = '/home/nadiafigueroa/dev/MATLAB/michaelchughes-NPBayesHMM/code/demo/segmentation_Bidan/mat_joint_BIDAN/';
s = what(dir);
matfiles=s.mat;

for a=1:numel(matfiles)
   X = [];
   matfilename = char(matfiles(a)); 
   load(matfilename)
   X(1,:) = fringerforce;
   X(2:7,:) = forcetorque';
   X(8:30,:) = jointAngles';
   X(31,:) = angle(:,3)';
   X(32,:) = time;
   Xn{a,1} = X; 
end

%% manual segmentation bidan 
dir = '/home/nadiafigueroa/dev/MATLAB/michaelchughes-NPBayesHMM/code/demo/segmentation_Bidan/mannualseg';
s_man = what(dir);
matfiles_man=s_man.mat;

for a=1:numel(matfiles_man)
   Xproc = [];
   matfilename = char(matfiles_man(a)); 
   load(matfilename)
   Xproc(1,:) = angle;
   Xproc(2,:) = force;
   Xproc(3,:) = torque;
   Xproc(4,:) = time;
   Xn_proc{a,1} = Xproc;
   Xn_segs{a,1} = seg;
end
%% Visualize raw data
exp = 10;
Xtest = Xn{exp,1};
name = strcat('Bottle Cap Opening Trial ', num2str(exp));
plotBottleCapData(Xtest(1,:), Xtest(2:7,:)', Xtest(8:30,:)', Xtest(31,:)', name);

%% Visualize processed data by Bidan
exp=10;
exp_b = exp-3;
Xtest_proc = Xn_proc{exp_b,1};
segtest  = Xn_segs{exp_b,1};
name = strcat('Bottle Cap Opening (Processed Data) Trial ', num2str(exp));
plotBottleCapDataBidan(Xtest_proc(1,:), Xtest_proc(2,:), Xtest_proc(3,:), name, segtest);

%% Visualize segmentation on data
exp = 10;
plotSegBottleCapData(Xn, bestGauPsi, exp);

%% Visualize segmentation on data with manual seg
timestamps_proc = Xtest_proc(4,:);
timestamps_raw = Xtest(32,:);

for i=1:length(segtest)
    sr_in = find(timestamps_raw == timestamps_proc( segtest(i,1)));
    sr_end = find(timestamps_raw == timestamps_proc( segtest(i,2)));
    segs_raw(i,:) = [sr_in sr_end];
end
plotSegBottleCapDataMan(Xn, bestGauPsi, exp, segs_raw);

%% Preprocess for segmentation (From MoCap data pre-processing experiment)

RobotChannelNames = {'Object.Normalforce', 'Object.fx', 'Object.fy', ... 
    'Object.fz', 'Object.tx', 'Object.ty','Object.tz','Hand.j1','Hand.j2',...
    'Hand.j3','Hand.j4','Hand.j5','Hand.j6','Hand.j7','Hand.j8','Hand.j9',...
    'Hand.j10','Hand.j11','Hand.j12','Hand.j13','Hand.j14','Hand.j15','Hand.j16',...
    'Hand.j17','Hand.j18','Hand.j19','Hand.j20','Hand.j21','Hand.j22','Hand.j23'};

N = length(Xn);
N = 18;
% nDim = 30;
nDim = 2;
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
for ii = 1:18
    
    % All values
    D = Xn{ii}(1:30,:);   
    
    % Normalize dimensions
    for jj=1:7
        Djj_n = D(jj,:)/range(D(jj,:));
        Djj_n(find(isnan(Djj_n)==1)) = 0;
        D(jj,:) =  Djj_n; 
    end

    % Use just normal force and tz as bidan
    D = D([1 7],:);
    
    % Enforce zero-mean
%     D = D';
%     D = bsxfun( @minus, D, mean(D,2) );
%     D = D';
   

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
