% Creates the 6 sequence motion capture dataset
%  from subjects 13 and 14 of the CMU database
clear all
close all
clc
%% Extract data from directory

d = '../data/CMU_Mocap/';
files = dir(strcat(d,'dat/','*.dat'));

ex_session = {};
for ii=1:length(files)
    filename = strcat(d,'dat/',files(ii,1).name);    
    clear data_
    data_ = textread(filename);
    ex_session{ii,1} = data_(:,1:12)';
end


%% Preprocess for segmentation (From MoCap data pre-processing experiment)
MocapChannelNames = {'root.ty', 'lowerback.rx', 'lowerback.ry', 'upperneck.ry', ...
    'rhumerus.rz', 'rradius.rx','lhumerus.rz', 'lradius.rx', ...
    'rtibia.rx', 'rfoot.rx', 'ltibia.rx', 'lfoot.rx'...
    };

Preproc.nObj = 6;
Preproc.obsDim = 12;
Preproc.R = 1;
Preproc.windowSize = 12;
Preproc.channelNames = MocapChannelNames;

GT = load( fullfile( d, 'mat', 'ExerciseGroundTruth.mat' ) );
GT = GT.GT;

% Create data structure
MocapData = SeqData();


for ii = 1:length(ex_session)
    
    D = ex_session{ii,1};
    
    % Enforce zero-mean, and apply block-averaging
     D = D';
     D = bsxfun( @minus, D, mean(D,1) );
     D = D';
        
    seqName = files.name;
    
    % Add the sequence to our data structure
    MocapData = MocapData.addSeq( D, seqName,  GT(ii).true_labels );
end
fprintf( '... read in all %d sequences\n', MocapData.N );

% -------------------------------------- Preproc autoregressive data
% This step simply builds necessary data structs for efficient AR inference
%   including filling in XprevR data field so that for any time tt,
%       XprevR(:,tt) = [Xdata(:, tt-1); Xdata(:,tt-2); ... Xdata(:,tt-R) ]
%   allowing of course for discontinuities at sequence transitions
MocapDataAR = ARSeqData( Preproc.R, MocapData);
