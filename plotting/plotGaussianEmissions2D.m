function [h] = plotGaussianEmissions2D(gmm, plot_labels, title_name, varargin)


font_size   = 18;

h = figure('Color',[1 1 1]);
colored = 0;
color_labels = 0;
if ~isempty(varargin)
    colored = 1;
    clear color_labels 
    color_labels = varargin{1};
    
end

if isfield(gmm,'K')
    K = gmm.K;
    gmm.Priors = ones(1,K)/K;
else
    K = length(gmm.Priors);
end

% Clustered Colors
if colored
    if color_labels==0
        colors= vivid(K);
    else
%         level = 10; n = ceil(level/2);
%         cmap1 = [linspace(1, 1, n); linspace(0, 1, n); linspace(0, 1, n)]';
%         cmap2 = [linspace(1, 0, n); linspace(1, 0, n); linspace(1, 1, n)]';
%         cmap = [cmap1; cmap2(2:end, :)];
%                
%         cmap_vivid = vivid(cmap, [.85, .85]);        
%         color_int = floor(length(cmap_vivid)/length(color_labels));        
%         color_range = 1:color_int:length(cmap_vivid)
%         for k =1:length(color_labels)
%             colors(k,:) = cmap_vivid(color_range(color_labels(k)),:);
%         end         
        color_range = hsv(max(color_labels));
        for k =1:length(color_labels)
            colors(k,:) = color_range(color_labels(k),:);
        end      
%         colormap(cmap);
       
    end        
else
    % Gray Color
    colors = repmat([0.3    0.3    0.3],[K,1]);
end


for i=1:K   
    hold on
      if color_labels==0
            plotGMM(gmm.Mu(:,i), gmm.Sigma(:,:,i), colors(i,:),1); 
            alpha(0.2)
      else
          plotGMM(gmm.Mu(:,i), gmm.Sigma(:,:,i), colors(i,:),1,1); 
      end
end
hold on;
ml_plot_centroid(gmm.Mu',colors); hold on;
ml_plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,colors,1); 
alpha 0.3;


% if isempty(varargin)
    labels = 1:K;
% else
%     labels = varargin{1};
% end

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