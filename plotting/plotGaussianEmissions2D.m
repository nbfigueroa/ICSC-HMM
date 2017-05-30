function [h] = plotGaussianEmissions2D(gmm, plot_labels, title_name, varargin)


font_size   = 18;

h = figure('Color',[1 1 1]);
colored = 0;

if isfield(gmm,'K')
    K = gmm.K;
    gmm.Priors = ones(1,K)/K;
else
    K = length(gmm.Priors);
end

% Clustered Colors
if colored
   colors= vivid(K);
else
    % Gray Color
    colors = repmat([0.3    0.3    0.3],[K,1]);
end

for i=1:K   
    hold on
    plotGMM(gmm.Mu(:,i), gmm.Sigma(:,:,i), colors(i,:),1); 
    alpha(0.3)
end
hold on;
ml_plot_centroid(gmm.Mu',colors); hold on;
ml_plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,colors,1); 


if isempty(varargin)
    labels = 1:K;
else
    labels = varargin{1};
end

for i=1:K   
    hold on;
    legend = sprintf('\\theta_{%d}',[labels(i)]);
    text(gmm.Mu(1,i) + gmm.Sigma(1,1,i) ,gmm.Mu(2,i) + gmm.Sigma(2,2,i),legend,'Interpreter','Tex','FontSize',font_size);   
end


xlabel(plot_labels{1},'Interpreter','Latex', 'FontSize',font_size,'FontName','Times', 'FontWeight','Light');            
ylabel(plot_labels{2},'Interpreter','Latex','FontSize',font_size,'FontName','Times', 'FontWeight','Light');
title (title_name,'Interpreter','Latex','FontSize',font_size,'FontName','Times', 'FontWeight','Light');          
axis equal
grid on; box on;


end