function [ ] = plotLabeledData( X, t , titlename, legends)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[D  N] = size(X);

if isempty(t)
    t = 1:N;
end

M = 10;
begin_time = t(1); 
xs = 1:N;    

% h = figure('Color', [1 1 1]);

ys = linspace(min(min(X(1:end,:))), max(max(X(1:end,:))),M);
imagesc(xs,ys,X(D,:),'CDataMapping','scaled');hold on;
alpha(0.3)
plot(t - begin_time, X(1:end-1,:)','-.', 'LineWidth',2);hold on;
axis( [1 N ys(1) ys(end)] );
xlabel('Time (1,...,T)')
ylabel('$\mathbf{x}$','Interpreter','LaTex')
legend(legends)
set(gca,'YDir','normal')
title(titlename,'Interpreter','LaTex','Fontsize',20)

end

