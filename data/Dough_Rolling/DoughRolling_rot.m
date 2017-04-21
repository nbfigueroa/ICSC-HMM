%% New Rolls
clear all
load individual_rolls.mat
figure('Name', 'Dough Rolling','Color',[1 1 1])
for ii=1:length(Xn)
    X = Xn{ii};
    scatter3(X(1,:), X(2,:), X(3,:),5,[rand rand rand],'filled');
    hold on
end
xlabel('x');ylabel('y');zlabel('z');
grid on
axis equal

%% Visualize Trajectories + FT
figure('Color',[1 1 1])
n = 4;
subplot(4,1,1)
plot(Xn{n,1}(1:3,:)')
legend('x','y','z')
subplot(4,1,2)
plot(Xn{n,1}(4:7,:)')
legend('qw','qi','qj', 'qk')
subplot(4,1,3)
plot(Xn{n,1}(8:10,:)') 
legend('fx','fy','fz')
subplot(4,1,4)
plot(Xn{n,1}(11:end,:)') 
legend('tx','ty','tz')

%% Check Rotation + Smooth data + Sub-sample
Xn_ch = [];
for jj=1:length(Xn)   
    
    % Check Rotation
    X_test = Xn{jj,1}; rot_end = 7;    
    X_new = X_test;    
    Q_new = checkRotations(X_test(4:rot_end,:));
    X_tmp = cat(1,X_test(1:3,:),Q_new);
    X_new = cat(1,X_tmp,X_test(rot_end+1:end,:));
    
    % Smooth data
    X_tmp = X_new;
    for kk=1:size(X_new,1)
        X_tmp(kk,:) = smooth(X_new(kk,:),round(size(X_tmp,2)*0.005),'moving');
    end
    X_new = X_tmp;
    
    % Scale Forces/Torques
    X_new(8:10,:) = X_tmp(8:10,:)/15;
    X_new(11:end,:) = X_tmp(11:end,:)/5;
    
    % Sub-sample data
    Xn_ch{jj,1} = X_new(:,1:5:end);
end

%% Visualize Pre-processed Trajectories + FT
figure('Color',[1 1 1])
n = 5;
subplot(4,1,1)
plot(Xn_ch{n,1}(1:3,:)')
legend('x','y','z')
subplot(4,1,2)
plot(Xn_ch{n,1}(4:7,:)')
legend('qw','qi','qj', 'qk')
subplot(4,1,3)
plot(Xn_ch{n,1}(8:10,:)') 
legend('fx','fy','fz')
subplot(4,1,4)
plot(Xn_ch{n,1}(11:end,:)') 
legend('tx','ty','tz')

%% Preprocess for segmentation (From MoCap data pre-processing experiment)
 
RobotChannelNames = {'pos.x','pos,y','pos.z', 'pos.qw','pos.qi','pos.qj','pos.qz'};


% Visualize 3d trajectories different color is different timeseries
XnT = Xn_ch;
N = length(Xn_ch);
nDim = 3;
Preproc.nObj = N;
Preproc.obsDim = nDim;
Preproc.R = 1;
Preproc.windowSize = nDim;
Preproc.channelNames = RobotChannelNames;

% Create data structure
Robotdata = SeqData();
RobotdataN = SeqData();


figure('Color', [1 1 1])
% Read in every sequence, one at a time
for ii = 1:N

    % Pos + Or + F + T
    D = Xn_ch{ii}(1:3,:);  
    
    % Enforce zero-mean
    D = D';
    D = bsxfun( @minus, D, mean(D,1) );
    D = D';
    
    % plot
    scatter3(D(1,:),D(2,:),D(3,:),5, [rand rand rand],'filled')
    hold on
    
    % Add the sequence to our data structure
    Robotdata = Robotdata.addSeq( D, num2str(ii));
   
end

grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z')
