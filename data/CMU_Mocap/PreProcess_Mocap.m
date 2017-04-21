% Creates the 6 sequence motion capture dataset
%  from subjects 13 and 14 of the CMU database
clear all
close all
clc
%% Extract data from directory

d = '/home/nadiafigueroa/dev/MATLAB/tGau-BP-HMM/data/CMU_Mocap/';
files = dir(strcat(d,'dat/','*.dat'));

data = {};
for ii=1:length(files)
    filename = strcat(d,'dat/',files(ii,1).name);    
    data_ = textread(filename);
    data{ii,1} = data_(:,1:12)';
end

data_length = size(data{ii,1},2);


%%


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
data = SeqData();

% Read in every sequence, one at a time
fprintf( 'Reading %d AMC files. Will take a long time...\n', length( myFileList) );
for ii = 1:length( myFileList )
    
    D = 
    
    % Enforce zero-mean, and apply block-averaging
    D = D';
    D = bsxfun( @minus, D, mean(D,2) );
    D = preprocessMocapData(D, Preproc.windowSize );
    
    % Drop the extension ".amc" from the current sequence name string
    [~,seqName,~] = fileparts( fname );
    
    % Add the sequence to our data structure
    data = data.addSeq( D, seqName,  GT(ii).true_labels );
end
fprintf( '... read in all %d sequences\n', data.N );

