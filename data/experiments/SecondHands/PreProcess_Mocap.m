%% [PRE_PROCESSING] Second Hands Ck-12 Environment Mocap Action Data 

clear all
close all
clc

%% Extract data from directory

d = '../data/SecondHands_Actions/bvh/';
files = dir(strcat(d,'*.bvh'));

actions = {};
for i=1:length(files)
    filename  =  files(i).name;
    
    % Extract Mocap Data
    [skel, channels, frameLength] = bvhReadFile(strcat(d, '/', filename));
    actions{i,1}.skel        = skel;
    actions{i,1}.channels    = channels;
    actions{i,1}.frameLength = frameLength;
end

% FUCK THIS SHIT
%     ||
%     ||
%     ||
%    \  /
%     \/
MocapChannelNames = {'hips.tx', 'hips.ty','hips.tz', 'hips.ry', 'hips.rx','hips.rz', ...
    'chest.ry', 'chest.rx','chest.rz', 'chest2.ry', 'chest2.rx','chest2.rz','chest3.ry', 'chest3.rx','chest3.rz',... 
    'chest4.ry', 'chest4.rx','chest4.rz','neck.ry', 'neck.rx','neck.rz','head.ry', 'head.rx','head.rz', ...
    'RightCollar.ry', 'RightCollar.rx','RightCollar.rz', 'RightShoulder.ry', 'RightShoulder.rx','RightShoulder.rz', ...
    'RightElbow.ry', 'RightElbow.rx','RightElbow.rz', 'RightWrist.ry', 'RightWrist.rx','RightWrist.rz', ...
    'LeftCollar.ry', 'LeftCollar.rx','LeftCollar.rz', 'LeftShoulder.ry', 'LeftShoulder.rx','LeftShoulder.rz', ...
    'LeftElbow.ry', 'LeftElbow.rx','LeftElbow.rz', 'LeftWrist.ry', 'LeftWrist.rx','LeftWrist.rz', ...
    'RightHip.ry','RightHip.rx','RightHip.rz', 'RightKnee.ry','RightKnee.rx','RightKnee.rz', ...
    'RightAnkle.ry','RightAnkle.rx','RightAnkle.rz', 'RightToe.ry','RightToe.rx','RightToe.rz', ...
    'LeftHip.ry','LeftHip.rx','LeftHip.rz', 'LeftKnee.ry','LeftKnee.rx','LeftKnee.rz', ...
    'LeftAnkle.ry','LeftAnkle.rx','LeftAnkle.rz', 'LeftToe.ry','LeftToe.rx','LeftToe.rz'    
    };


%% Play Mocap Data Sample (if you want)

% Choose an action
action = actions{2};

% Play it
skelPlayData(action.skel, action.channels, action.frameLength);


%% Raw Dataset
figure;
plot(actions{1}.channels)

%% Extract Relevant Joints and Angles

%%%%%% Ideal Example of Mocap Data from CMU Dataset (12 D in total!!)%%%%%
%     MocapChannelNames = {'root.ty', 'lowerback.rx', 'lowerback.ry', ...
%     'upperneck.ry','rhumerus.rz', 'rradius.rx','lhumerus.rz', ...
%     'lradius.rx','rtibia.rx', 'rfoot.rx', 'ltibia.rx', 'lfoot.rx'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     

clc;
%%% Choose Subset of Joints from Mocap Data (11 Joints = 33D)
SkelJoints = {'Hips','Head', 'RightShoulder','RightElbow', 'RightWrist', ...
              'LeftShoulder','LeftElbow', 'LeftWrist', ...
              'RightHip', 'RightKnee', 'LeftHip', 'LeftKnee'};

for ii = 1:length(actions);
  skeleton = actions{ii}.skel.tree;
  joint_dim = [];
  for jj = 1:length(SkelJoints)
      joint_name = SkelJoints{jj};
      for kk = 1:length(skeleton)
          if strcmp(joint_name,skeleton(kk).name)         
              if strcmp(joint_name,'Hips')
                  joint_dim = [joint_dim skeleton(kk).posInd];                  
              else
                  joint_dim = [joint_dim skeleton(kk).rotInd];              
              end
          end
      end
  end
  
  % Choose joint angles (Subsample 1/5)
  D_tmp = actions{ii}.channels(1:1:end,joint_dim(:))';
  
  % Scale Translational components
  D_tmp(1:3,:) = D_tmp(1:3,:)/100;
  
  % Convert Rotational Components to Radians
  D_tmp(4:end,:) = D_tmp(4:end,:)*(pi/180);
  
  % Smooth Data and Check Rotation Discontinuities
  for jj=1:size(D_tmp,1)      
     D_tmp(jj,:) = smooth(D_tmp(jj,:),round(size(D_tmp,2)*0.01),'moving');          
     if jj > 3
        D_tmp(jj,:) = checkRotation(D_tmp(jj,:));
     end
  end
  
  actions{ii}.sel_joints = D_tmp;
end


%% Visualize Processed Data of each Action

close all
for ii = 1:length(actions)
    
    action = actions{ii};
    
    figure('Color', [1 1 1])
    subplot(5,1,1)
    plot(action.sel_joints(1:6,:)','--','LineWidth', 1.5)
    legend('Hips-tranx','Hips-trany','Hips-tranz','Head-roty','Head-rotx','Head-rotz')
    grid on 
    box on
    axis tight
    
    subplot(5,1,2)
    plot(action.sel_joints(7:15,:)','--','LineWidth', 1.5)
    legend('RS-roty','RS-rotx','RS-rotz','RE-roty','RE-rotx','RE-rotz', 'RW-roty','RW-rotx','RW-rotz')
    grid on 
    box on
    axis tight
    
    subplot(5,1,3)
    plot(action.sel_joints(16:24,:)','--','LineWidth', 1.5)
    legend('LS-roty','LS-rotx','LS-rotz', 'LE-roty','LE-rotx','LE-rotz','LW-roty','LW-rotx','LW-rotz')
    grid on 
    box on
    axis tight
    
    subplot(5,1,4)
    plot(action.sel_joints(22:27,:)','--','LineWidth', 1.5)
    legend('RH-roty','RH-rotx','RH-rotz', 'RK-roty','RK-rotx','RK-rotz')
    grid on 
    box on
    axis tight
    
    subplot(5,1,5)
    plot(action.sel_joints(28:33,:)','--','LineWidth', 1.5)
    legend('LH-roty','LH-rotx','LH-rotz', 'LK-roty','LK-rotx','LK-rotz')
    grid on 
    box on
    axis tight
    
    [~,seqName,~] = fileparts( files(ii).name );
    
    suptitle(seqName)
end


%% Preprocess for segmentation (From MoCap data pre-processing experiment)

Preproc.nObj = length(actions);
Preproc.nDim = length(joint_dim);
Preproc.R = 1;
Preproc.windowSize = 12;
Preproc.channelNames = MocapChannelNames;

% Create data structure
MocapData = SeqData();

% Read in every sequence, one at a time
for ii = 1:length( actions )
    
    % Read Joint Angles from Data Structure 
    D = actions{ii}.sel_joints;
    
    % Enforce zero-mean
    D = D';
    D = bsxfun( @minus, D, mean(D,2) );
    D = D';
%     D = preprocessMocapData(D, Preproc.windowSize, Preproc.nDim);
    
        
    % Drop the extension ".amc" from the current sequence name string
    [~,seqName,~] = fileparts( files(ii).name );
    
    % Add the sequence to our data structure
    MocapData = MocapData.addSeq( D, seqName );
    
end
fprintf( '... read in all %d sequences\n', MocapData.N );


% -------------------------------------- Preproc autoregressive data
% This step simply builds necessary data structs for efficient AR inference
%   including filling in XprevR data field so that for any time tt,
%       XprevR(:,tt) = [Xdata(:, tt-1); Xdata(:,tt-2); ... Xdata(:,tt-R) ]
%   allowing of course for discontinuities at sequence transitions
MocapDataAR = ARSeqData( Preproc.R, MocapData);
