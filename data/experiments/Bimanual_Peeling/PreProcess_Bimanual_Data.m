% Prepares Bimanual Data for Segmentation
clear all
close all
clc

%% Load Data from Mat File
data_dir = './data/';
load(strcat(data_dir,'proc_data.mat'))
N = length(proc_data);

% Plot Trajectories with Reference Frames etc
for ii=1:N
    
    plotEEData( proc_data{ii}.active.X, proc_data{ii}.active.t ,strcat(proc_data{ii}.name,': Active Arm'))
    plotEEData( proc_data{ii}.passive.X, proc_data{ii}.passive.t ,strcat(proc_data{ii}.name,': Passive Arm'))
%     plotObjectData( proc_data{ii}.object.feats, proc_data{ii}.active.t, strcat(proc_data{ii}.name,': Object'))
    plotBimanualTrajectories(proc_data{ii}); 

end
%% Move to tGau-BP-HMM Folder
tgau_dir = '/home/nbfigueroa/dev/tGau-BP-HMM/experiments/Bimanual_Peeling';
cd(tgau_dir)

%% Preprocess for segmentation (Do BP-HMM on joint arm data + object Features)

ChannelNames = {};

Preproc.nObj   = N;
Preproc.obsDim = 32;
Preproc.channelNames = ChannelNames;

% Create data structure
Bimanual_Arm_Data  = SeqData();

force_sc = 15;
color_sc = 80;

for ii = 1:N
    
    % %%%% Extract Data for Active Arm %%%%%
    X_a = proc_data{ii}.active.X;
    
    % Scale Variables 
    D_a = X_a;
    D_a(1:3,:)    = X_a(1:3,:)/1.5;
    D_a(4:7,:)    = X_a(4:7,:)/2;
    D_a(8:10,:)   = X_a(8:10,:)/force_sc;
    D_a(11:end,:) = X_a(11:end,:)/2;
    
    % Enforce zero-mean, and apply block-averaging
     D_a = D_a';
     D_a = bsxfun( @minus, D_a, mean(D_a,2) );
     D_a = D_a';       
    
    % %%%% Extract Data for Passive Arm %%%%%
    X_p = proc_data{ii}.passive.X;
    
    % Scale Variables 
    D_p = X_p;
    D_p(1:3,:)    = X_p(1:3,:)/3;
    D_p(4:7,:)    = X_p(4:7,:);
    D_p(8:10,:)   = X_p(8:10,:)/(force_sc*1.5);
    D_p(11:end,:) = X_p(11:end,:)/2;
    
    % Enforce zero-mean, and apply block-averaging
     D_p = D_p';
     D_p = bsxfun( @minus, D_p, mean(D_p,2) );
     D_p = D_p';       
    
    % %%%% Extract Data for Object %%%%%
    X_o = proc_data{ii}.object.feats;
    
    % Scale Variables 
    D_o = X_o;
    D_o(1:3,:)    = X_o(1:3,:)/color_sc;    
    D_o(4:end,:)  = X_o(4:end,:)/(color_sc/2);
    
    % Enforce zero-mean, and apply block-averaging
    D_o = D_o';
    D_o = bsxfun( @minus, D_o, mean(D_o,2) );
    D_o = D_o';                 
     
    % Add the sequence to our data structure
    D = [D_a(:,1:length(D_p));D_p; D_o(:,1:length(D_p))];
    Bimanual_Arm_Data = Bimanual_Arm_Data.addSeq( D, proc_data{ii}.name);
        
end
fprintf( '... read in all %d sequences\n', Bimanual_Arm_Data.N );

%% Preprocess for segmentation (Do BP-HMM on each arm independently)

ChannelNames = {'t.x', 't.y', 't.z', 'q.w', 'q.i', 'q.j', 'q.k', ... 
    'f.x', 'f.y', 'f.z', 'tau.x', 'tau.y', 'tau.z' };

Preproc.nObj   = N;
Preproc.obsDim = 13;
Preproc.channelNames = ChannelNames;

% Create data structure
Active_Arm_Data  = SeqData();
Passive_Arm_Data = SeqData();

force_sc = 15;

for ii = 1:N
    
    % %%%% Extract Data for Active Arm %%%%%
    X = proc_data{ii}.active.X;
    
    % Scale Variables 
    clear D; D = X;
    D(1:3,:)    = X(1:3,:)/1.5;
    D(4:7,:)    = X(4:7,:)/2;
    D(8:10,:)   = X(8:10,:)/force_sc;
    D(11:end,:) = X(11:end,:)/2;
    
    % Enforce zero-mean, and apply block-averaging
     D = D';
     D = bsxfun( @minus, D, mean(D,2) );
     D = D';       
  
    % Add the sequence to our data structure
    Active_Arm_Data = Active_Arm_Data.addSeq( D, proc_data{ii}.name);
    
    % %%%% Extract Data for Passive Arm %%%%%
    clear X;
    X = proc_data{ii}.passive.X;
    
    % Scale Variables 
    clear D; D = X;
    D(1:3,:)    = X(1:3,:)/2;
    D(4:7,:)    = X(4:7,:)*2;
    D(8:10,:)   = X(8:10,:)/(force_sc);
    D(11:end,:) = X(11:end,:)/2;

    % Enforce zero-mean, and apply block-averaging
     D = D';
     D = bsxfun( @minus, D, mean(D,2) );
     D = D';       
    
    % Add the sequence to our data structure
    Passive_Arm_Data = Passive_Arm_Data.addSeq( D, proc_data{ii}.name);
        
    
end
fprintf( '... read in all %d sequences\n', Active_Arm_Data.N );

%% Preprocess for segmentation (Do BP-HMM on joint arm data)

ChannelNames = {'t.x', 't.y', 't.z', 'q.w', 'q.i', 'q.j', 'q.k', ... 
    'f.x', 'f.y', 'f.z', 'tau.x', 'tau.y', 'tau.z' };

Preproc.nObj   = N;
Preproc.obsDim = 26;
Preproc.channelNames = ChannelNames;

% Create data structure
Bimanual_Arm_Data  = SeqData();

force_sc = 15;

for ii = 1:N
    
    % %%%% Extract Data for Active Arm %%%%%
    X_a = proc_data{ii}.active.X;
    
    % Scale Variables 
    D_a = X_a;
    D_a(1:3,:)    = X_a(1:3,:)/1.5;
    D_a(4:7,:)    = X_a(4:7,:)/2;
    D_a(8:10,:)   = X_a(8:10,:)/force_sc;
    D_a(11:end,:) = X_a(11:end,:)/2;
    
    % Enforce zero-mean, and apply block-averaging
     D_a = D_a';
     D_a = bsxfun( @minus, D_a, mean(D_a,2) );
     D_a = D_a';       
  
    
    % %%%% Extract Data for Passive Arm %%%%%
    X_p = proc_data{ii}.passive.X;
    
    % Scale Variables 
    D_p = X_p;
    D_p(1:3,:)    = X_p(1:3,:)/3;
    D_p(4:7,:)    = X_p(4:7,:);
    D_p(8:10,:)   = X_p(8:10,:)/(force_sc*1.5);
    D_p(11:end,:) = X_p(11:end,:)/2;

    % Enforce zero-mean, and apply block-averaging
     D_p = D_p';
     D_p = bsxfun( @minus, D_p, mean(D_p,2) );
     D_p = D_p';       
    
    % Add the sequence to our data structure
    D = [D_a(:,1:length(D_p));D_p];
    Bimanual_Arm_Data = Bimanual_Arm_Data.addSeq( D, proc_data{ii}.name);
        
end
fprintf( '... read in all %d sequences\n', Active_Arm_Data.N );

