function [] = plotBottleCapData(fingerforce,forcetorque,jointAngles,angle,name)
% Create figure
figure1 = figure('Name',name,'Color',[1 1 1]);
M = 10;

% Create subplot
subplot1 = subplot(5,1,1,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot1,'on');
hold(subplot1,'all');
% Create plot
plot(fingerforce,'Parent',subplot1,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
% ylabel('Finger force','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
title('Finger Force')
ys = linspace(min(min(fingerforce)), max(max(fingerforce)),M);
axis( [1 length(fingerforce) ys(1) ys(end)] );


% Create subplot
subplot2 = subplot(5,1,2,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot2,'on');
hold(subplot2,'all');
% Create plot
plot(forcetorque(:,1:3),'Parent',subplot2,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
% ylabel('Cap Force/torque','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
title('Bottle cap Force')
legend('fx','fy','fz');
ys = linspace(min(min(forcetorque(:,1:3))), max(max(forcetorque(:,1:3))),M);
axis( [1 length(forcetorque(:,1:3)) ys(1) ys(end)] );

subplot3 = subplot(5,1,3,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot3,'on');
hold(subplot3,'all');
% Create plot
plot(forcetorque(:,4:6),'Parent',subplot3,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
% ylabel('Cap Force/torque','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
title('Bottle cap Torque')
legend('tx','ty','tz');
ys = linspace(min(min(forcetorque(:,4:6))), max(max(forcetorque(:,4:6))),M);
axis( [1 length(forcetorque(:,4:6)) ys(1) ys(end)] );


% Create subplot
subplot4 = subplot(5,1,4,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot4,'on');
hold(subplot4,'all');
% Create plot
plot(jointAngles,'Parent',subplot4,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
% ylabel('Joint angles (23)','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
title('Joint Angles')
ys = linspace(min(min(jointAngles)), max(max(jointAngles)),M);
axis( [1 length(jointAngles) ys(1) ys(end)] );

% % Create subplot
subplot5 = subplot(5,1,5,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot5,'on');
hold(subplot5,'all');
% Create plot
plot(angle,'Parent',subplot5,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% ylabel('Cap rotation angles','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
title('Bottle Cap rotation angle (Z)')
ys = linspace(min(min(angle)), max(max(angle)),M);
axis( [1 length(angle) ys(1) ys(end)] );

% % Create subplot
% subplot4 = subplot(6,1,4,'Parent',figure1,'YGrid','on','FontSize',12);
% box(subplot4,'on');
% hold(subplot4,'all');
% % Create plot
% plot(trackWrist,'Parent',subplot4,'LineWidth',2);
% % Create xlabel
% xlabel('samples','FontSize',12);
% % Create ylabel
% % ylabel('Wrist position','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% title('Wrist position')
% legend('x','y','z');
% 
% % Create subplot
% subplot5 = subplot(6,1,5,'Parent',figure1,'YGrid','on','FontSize',12);
% box(subplot5,'on');
% hold(subplot5,'all');
% % Create plot
% plot(trackCap,'Parent',subplot5,'LineWidth',2);
% % Create xlabel
% xlabel('samples','FontSize',12);
% % Create ylabel
% % ylabel('Bottle Cap Position','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% title('Bottle Cap Position')
% legend('x','y','z');
% 

end