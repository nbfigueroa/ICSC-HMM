function [ ] = plotLabeledEEData( X, t , titlename, full,legends)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[D  N] = size(X);

if isempty(t)
    t = 1:N;
end
M = 10;
begin_time = t(1); 
xs = 1:N;    

figure('Color', [1 1 1]);

if (full)        
    subplot(4,1,1);
    ys = linspace(min(min(X(1:3,:))), max(max(X(1:3,:))),M);
    imagesc(xs,ys,X(D,:),'CDataMapping','scaled');hold on;
    alpha(0.3)
    plot(t - begin_time, X(1:3,:)','-.', 'LineWidth',2);hold on;
    axis( [1 N ys(1) ys(end)] );
    legend('x','y','z')
    ylabel('Position [m]')
    set(gca,'YDir','normal')
    
    f_start = 8;
    t_start = 11;
    subplot(4,1,2);
    if D == 14
        ys = linspace(min(min(X(4:7,:))), max(max(X(4:7,:))),M);
        imagesc(xs,ys,X(D,:));hold on;
        alpha(0.3)
        plot(t - begin_time, X(4:7,:)','-.', 'LineWidth',2);
        ylabel('Orientation [.]')
        axis( [1 N ys(1) ys(end)] );
        legend('q_w','q_i', 'q_j', 'q_k')
        set(gca,'YDir','normal')
    else
        ys = linspace(min(min(X(4:6,:))), max(max(X(4:6,:))),M);
        imagesc(xs,ys,X(D,:));hold on;
        alpha(0.3)
        plot(t - begin_time, X(4:6,:)','-.', 'LineWidth',2);
        axis( [1 N ys(1) ys(end)] );
        ylabel('Orientation [deg]')
        legend('roll','pitch','yaw')
        set(gca,'YDir','normal')
        f_start = f_start - 1;
        t_start = t_start - 1;
    end
    
    subplot(4,1,3);
    ys = linspace(min(min(X(f_start:f_start+2,:))), max(max(X(f_start:f_start+2,:))),M);
    imagesc(xs,ys,X(D,:));hold on;
    alpha(0.3)
    plot(t - begin_time, X(f_start:f_start+2,:)','-.', 'LineWidth',2);
    axis( [1 N ys(1) ys(end)] );
    ylabel('Force [N]')
    legend('f_x','f_y', 'f_z')
    set(gca,'YDir','normal')
    
    subplot(4,1,4);
    ys = linspace(min(min(X(t_start:13,:))), max(max(X(t_start:13,:))),M);
    imagesc(xs,ys,X(D,:));hold on;
    alpha(0.3)
    plot(t - begin_time, X(t_start:13,:)','-.', 'LineWidth',2);
    axis( [1 N ys(1) ys(end)] );
    ylabel('Torque [Nm]')
    legend('\tau_x','\tau_y', '\tau_z')
    set(gca,'YDir','normal')    
    suptitle(titlename)
    
else
    ys = linspace(min(min(X(1:end,:))), max(max(X(1:end,:))),M);
    imagesc(xs,ys,X(D,:),'CDataMapping','scaled');hold on;
    alpha(0.3)
    plot(t - begin_time, X(1:end-1,:)','-.', 'LineWidth',2);hold on;
    axis( [1 N ys(1) ys(end)] );
%     legend('x','y','z','q_w','q_i', 'q_j', 'q_k','f_x','f_y', 'f_z','\tau_x','\tau_y', '\tau_z')
    legend(legends)
    set(gca,'YDir','normal')        
    title(titlename,'Interpreter','LaTex','Fontsize',20)
    
end

end

