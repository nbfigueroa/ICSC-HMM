%% Extract Rolling Data for Regression Practical
clear all
load Series3_Segmented.mat

ee_traj_reach = [];
ee_traj_roll = [];
ee_traj_back = [];

figure('Color',[1 1 1])
subplot(1,3,1)
for ii=1:length(Phase1)
    ee_traj = Phase1(1,ii).in_dough_interp.EE_POS;
    ee_vel = diff(ee_traj')';    
    ee_traj_reach = [ee_traj(:,2:end) ; ee_vel];
    ee_traj_reach = [ee_traj_reach ee_traj_reach]    
end

scatter3(ee_traj_reach(1,:),ee_traj_reach(2,:),ee_traj_reach(3,:),5,[rand rand rand])
title('Reaching Motions')
grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z')

%%
fileID = fopen('reach_demo_1.txt','w');
fprintf(fileID,'%6.2f,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f\n',data_2_s');
fclose(fileID);

subplot(1,3,2)
for ii=1:length(Phase2)
    ee_traj = Phase2(1,ii).in_dough_interp.EE_POS;
    ee_vel = diff(ee_traj')';
    ee_traj_roll{1,ii} = [ee_traj(:,2:end) ; ee_vel];    
    scatter3(ee_traj(1,:),ee_traj(2,:),ee_traj(3,:),5,[rand rand rand])     
    hold on    
end
title('Rolling Motions')
grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z')

subplot(1,3,3)
for ii=1:length(Phase3)
    ee_traj = Phase3(1,ii).in_dough_interp.EE_POS;
    ee_vel = diff(ee_traj')';
    ee_traj_back{1,ii} = [ee_traj(:,2:end) ; ee_vel];
    scatter3(ee_traj(1,:),ee_traj(2,:),ee_traj(3,:),5,[rand rand rand])
    hold on    
end
title('Retracting Motions')
grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z')

%% Extract Pouring Data for Regression Practical
clear all
load Pouring4Joel_All_Trials_Segmented.mat

ee_traj_reach = [];
ee_traj_back = [];

figure('Color',[1 1 1])
subplot(1,2,1)
for ii=1:length(Phase1)
    ee_traj = Phase1(1,ii).in_dough_interp.EE_POS;
    ee_vel = diff(ee_traj')';    
    ee_traj_reach{1,ii} = [ee_traj(:,2:end) ; ee_vel];
    scatter3(ee_traj(1,:),ee_traj(2,:),ee_traj(3,:),5,[rand rand rand])
    hold on    
end
title('Reaching Motions')
grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z')

subplot(1,2,2)
for ii=1:length(Phase2)
    ee_traj = Phase2(1,ii).in_dough_interp.EE_POS;
    ee_vel = diff(ee_traj')';            
    ee_traj_back{1,ii} = [ee_traj(:,2:end) ; ee_vel];
    scatter3(ee_traj(1,:),ee_traj(2,:),ee_traj(3,:),5,[rand rand rand])
    hold on    
end
title('Retracting Motions')
grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z')

