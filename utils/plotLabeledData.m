function [ ] = plotLabeledData( X, t , titlename, legends, label_range)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[D  N] = size(X);

if isempty(t)
    t = 1:N;
end

M = 10;
begin_time = t(1); 
xs = 1:N;    
ys = linspace(min(min(X(1:end,:))), max(max(X(1:end,:))),M);

% Without label_range
if isempty(label_range)
    imagesc(xs,ys,X(D,:),'CDataMapping','scaled'); hold on;
else  % With label_range  
    imagesc( xs, ys, repmat(X(D,:), M, 1), [1 max( label_range)] ); hold on;
end

alpha(0.3)
plot(t - begin_time, X(1:end-1,:)','-.', 'LineWidth',2);hold on;
axis( [1 N ys(1) ys(end)] );
xlabel('Time (1,...,T)')
ylabel('$\mathbf{x}$','Interpreter','LaTex')
legend(legends)
set(gca,'YDir','normal')
title(titlename,'Interpreter','LaTex','Fontsize',20)

end

