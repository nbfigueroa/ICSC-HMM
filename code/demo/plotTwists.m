function [] = plotTwists(vs, omegas, name)

% Create figure
figure1 = figure('Name',name,'Color',[1 1 1]);
% Create subplot
subplot1 = subplot(3,2,1,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot1,'on');
hold(subplot1,'all');
% Create plot
plot(omegas(1,:),'Parent',subplot1,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('\omega_x','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot2 = subplot(3,2,2,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot2,'on');
hold(subplot2,'all');
% Create plot
plot(vs(1,:),'Parent',subplot2,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('v_x','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot3 = subplot(3,2,3,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot3,'on');
hold(subplot3,'all');
% Create plot
plot(omegas(2,:),'Parent',subplot3,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('\omega_y','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot4 = subplot(3,2,4,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot4,'on');
hold(subplot4,'all');
% Create plot
plot(vs(2,:),'Parent',subplot4,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('v_y','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot5 = subplot(3,2,5,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot5,'on');
hold(subplot5,'all');
% Create plot
plot(omegas(3,:),'Parent',subplot5,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('\omega_z','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot6 = subplot(3,2,6,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot6,'on');
hold(subplot6,'all');
% Create plot
plot(vs(3,:),'Parent',subplot6,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('v_z','FontSize',16,'Rotation',0,'HorizontalAlignment','right');

suptitle('Twists');
end