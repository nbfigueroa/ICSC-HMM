function [ ] = plotLabeledObjectData( X, t , titlename, full)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[D N] = size(X);

if isempty(t)
    t = 1:N;
end
M = 10;
begin_time = t(1); 
xs = 1:N;    

figure('Color', [1 1 1]);
if (full)
    subplot(2,1,1);
    ys = linspace(min(min(X(1:3,:))), max(max(X(1:3,:))),M);
    imagesc(xs,ys,X(D,:),'CDataMapping','scaled');hold on;
    alpha(0.3)
    plot(t - begin_time, X(1:3,:)','-.', 'LineWidth',2);hold on;
    axis( [1 N ys(1) ys(end)] );
    ylabel('8-bit RGB Space')
    legend('\mu_r','\mu_g','\mu_b')
    set(gca,'YDir','normal')
    
    subplot(2,1,2);
    ys = linspace(min(min(X(4:6,:))), max(max(X(4:6,:))),M);
    imagesc(xs,ys,X(D,:));hold on;
    alpha(0.3)
    plot(t - begin_time, X(4:6,:)','-.', 'LineWidth',2);
    axis( [1 N ys(1) ys(end)] );
    ylabel('8-bit RGB Space')
    legend('\sigma_r','\sigma_g', '\sigma_b')
    set(gca,'YDir','normal')
    suptitle(titlename)
else
    
    ys = linspace(min(min(X(1:6,:))), max(max(X(1:6,:))),M);
    imagesc(xs,ys,X(D,:),'CDataMapping','scaled');hold on;
    alpha(0.3)
    plot(t - begin_time, X(1:6,:)','-.', 'LineWidth',2);hold on;
    axis( [1 N ys(1) ys(end)] );
    ylabel('8-bit RGB Space')
    legend('\mu_r','\mu_g','\mu_b','\sigma_r','\sigma_g', '\sigma_b')
    set(gca,'YDir','normal')
    title(titlename)

end

end

