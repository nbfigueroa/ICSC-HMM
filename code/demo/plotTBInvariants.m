function [] = plotTBInvariants(w1,v1,w2,v2,w3,v3,name)

%Plot timebased invariants
% Create figure
figure1 = figure('Name',name,'Color',[1 1 1]);
% Create subplot
subplot1 = subplot(3,2,1,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot1,'on');
hold(subplot1,'all');
% Create plot
plot(w1,'Parent',subplot1,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('\omega_1','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot2 = subplot(3,2,2,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot2,'on');
hold(subplot2,'all');
% Create plot
plot(v1,'Parent',subplot2,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('v_1','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot3 = subplot(3,2,3,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot3,'on');
hold(subplot3,'all');
% Create plot
plot(w2,'Parent',subplot3,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('\omega_2','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot4 = subplot(3,2,4,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot4,'on');
hold(subplot4,'all');
% Create plot
plot(v2,'Parent',subplot4,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('v_2','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot5 = subplot(3,2,5,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot5,'on');
hold(subplot5,'all');
% Create plot
plot(w3,'Parent',subplot5,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('\omega_3','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
% Create subplot
subplot6 = subplot(3,2,6,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot6,'on');
hold(subplot6,'all');
% Create plot
plot(v3,'Parent',subplot6,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
ylabel('v_3','FontSize',16,'Rotation',0,'HorizontalAlignment','right');

% Create subplot
suptitle('Time-based invariants');

end