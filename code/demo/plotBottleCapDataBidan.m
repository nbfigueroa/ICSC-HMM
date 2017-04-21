function [] = plotBottleCapDataBidan(angle,force,torque,name,seg)
% Create figure
figure1 = figure('Name',name,'Color',[1 1 1]);
M = 10;


% % Create subplot
subplot1 = subplot(3,1,1,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot1,'on');
hold(subplot1,'all');
% Create plot
plot(angle,'Parent',subplot1,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% ylabel('Cap rotation angles','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
title('Bottle Cap rotation angle (Z)')
ys = linspace(min(min(angle)), max(max(angle)),M);
axis( [1 length(angle) ys(1) ys(end)] );
yL = get(gca,'YLim');
for i=1:length(seg)
    seglims = seg(i,:);
    seg_in  = seglims(1);
    seg_end = seglims(2);
    c(i,:) = [rand rand rand];
    line([seg_in seg_in],yL,'Color',c(i,:),'LineWidth',3);
    line([seg_end  seg_end],yL,'Color',c(i,:),'LineWidth',3);
end

% Create subplot
subplot2 = subplot(3,1,2,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot2,'on');
hold(subplot2,'all');
% Create plot
plot(force,'Parent',subplot2,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
title('Finger Force')
ys = linspace(min(min(force)), max(max(force)),M);
axis( [1 length(force) ys(1) ys(end)] );
yL = get(gca,'YLim');
for i=1:length(seg)
    seglims = seg(i,:);
    seg_in  = seglims(1);
    seg_end = seglims(2);
    line([seg_in seg_in],yL,'Color',c(i,:),'LineWidth',3);
    line([seg_end  seg_end],yL,'Color',c(i,:),'LineWidth',3);
end



subplot3 = subplot(3,1,3,'Parent',figure1,'YGrid','on','FontSize',12);
box(subplot3,'on');
hold(subplot3,'all');
% Create plot
plot(torque,'Parent',subplot3,'LineWidth',2);
% Create xlabel
xlabel('samples','FontSize',12);
% Create ylabel
% ylabel('Cap Force/torque','FontSize',16,'Rotation',0,'HorizontalAlignment','right');
title('Bottle cap z-axis torque (tz)')
ys = linspace(min(min(torque)), max(max(torque)),M);
axis( [1 length(torque) ys(1) ys(end)] );
yL = get(gca,'YLim');
for i=1:length(seg)
    seglims = seg(i,:);
    seg_in  = seglims(1);
    seg_end = seglims(2);
    line([seg_in seg_in],yL,'Color',c(i,:),'LineWidth',3);
    line([seg_end  seg_end],yL,'Color',c(i,:),'LineWidth',3);
end

end