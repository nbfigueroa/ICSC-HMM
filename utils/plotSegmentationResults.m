function [Clustering Segmentation Total_feats hIM my_color_map] = plotSegmentationResults( data, Psi, iis, title_fig, groups )

max_feats = 0;
Zests = [];
for i=1:length(iis)
    ii = iis(i);
    Zest = Psi.stateSeq(1,ii).z;
    Zests = [Zests unique(Zest)];
    if  max(Zest)>max_feats
        max_feats = max(Zest);
    end
end

Total_feats = unique(Zests);
fprintf('Total Features: %d\n', length(Total_feats));

if length(Total_feats)==1
    max_feats = 1;
end

% if isempty(groups)
%    display('No feature grouping');    
%    number_plots = 2;
% else     
fprintf('Total Feature Groups: %d\n', length(groups)); 
number_plots = 3;
   
if isempty(groups)
    beh_label_plot = 1;
    emission_plot = 2;
    plot_gap = 0.05;
else    
    beh_label_plot = 2;
    emission_plot = 3;
    plot_gap = 0.05;           
end


% %%%%%%%%%% Segmentation Figure %%%%%%%%%%%%%%%
figure('Color',[1 1 1], 'Position',[ 1987 547 746  547]);
Segmentation = [];
for j=1:length(iis)
    subplot(length(iis),1,j)    
    ii = iis(j); 
    T = data.Ts(ii);
    M = 10;
    
    X = data.seq(ii);
    Zest = Psi.stateSeq(1,ii).z;
    
    xs = 1:T;    
    ys = linspace(min(min(X)), max(max(X)),M);
    hold all;
    
   
    segment_labels = repmat(Zest, M, 1);
    if ~isempty(groups) 
        Cest = Zest;
        cluster_labels = 1:length(groups);
        cluster_labels = cluster_labels + Total_feats(end);

        for k=1:length(groups)
            cluster_label = cluster_labels(k);
            group = groups{k};
            for kk=1:length(group)            
                Cest(find(Zest == group(kk))) = cluster_label;
            end               
        end
        cluster_labels = repmat(Cest, M/2, 1);
        segment_labels(ceil(M/2)+1:end,:) = cluster_labels; 
    end
    hIM = imagesc( xs, ys, segment_labels, [1 max_feats+length(groups)] );
    
    Y = get(hIM, 'CData');

    for i=1:size(X,1)
        if j==1
            c(i,:) = [rand rand rand];
        end
        plot( xs, X(i,:), 'Color',c(i,:), 'LineWidth', 1.5);
    end

    %Draw line at seg points
    [seg_results] = findSegPoints(Zest);
    Segmentation{ii,1} = seg_results;
%     yL = get(gca,'YLim');
%     yL_half = [ys(1)  median(ceil(ys(1)):1:ys(end))];
% 
%     for jj=1:size(seg_results,1)
%             model = seg_results(jj,1);       
%             seg_end = seg_results(jj,2);
%             line([seg_end seg_end],yL_half,'Color','r','LineWidth',3);
%             if jj==1
%                 start=0;
%             else
%                 start = seg_results(jj-1,2);
%             end
%             
%     end
    axis( [1 T ys(1) ys(end)] );
end
suptitle(title_fig);
%# make all text in the figure to size 14 and bold
figureHandle = gcf;
set(findall(figureHandle,'type','text'),'fontSize',15)  


% Second figure including Behavior Labels, Models and Groupings

if isfield(Psi, 'theta')

    if isempty(groups) 
        figure('Units', 'normalized','Color', [1 1 1], 'Position',[0.7107 0.4542  0.1445 0.4567])   
        beh_label_plot = 1;
        emission_plot = 2;
        plot_gap = 0.05;
    else    
        figure('Units', 'normalized','Color', [1 1 1], 'Position',[0.7107 0.4542  0.1445 0.4567])   
        beh_label_plot = 2;
        emission_plot = 3;
        plot_gap = 0.05;

        % %%%%%%%%%% Behavior Groups Figure %%%%%%%%%%%%%%%
        subplot(number_plots, 1, 1);
        y = [1 2];
        group_labels = 1:length(groups);
        group_labels = group_labels + Total_feats(end);

        % % Substitute features by their corresponding group labels
        group_feats = Total_feats;
        for k=1:length(groups)
            group_label = group_labels(k);
            group = groups{k};
            for j=1:length(group)            
                group_feats(find(Total_feats == group(j))) = group_label;
            end
        end

        hIM3 = imagesc(Total_feats, y, group_feats, [1 max_feats+length(groups)]);
        axis off
        axis equal tight
        axis([0 Total_feats(end)+1 1 2])
        for i=1:length(group_feats)
            behav = group_feats(i);
            y_axis = 1.5;
    %         text(Total_feats(i),y_axis,num2str(find(group_labels==behav)));
            text(Total_feats(i),y_axis,num2str(behav));
        end
        title('Cluster Labels')

    end

% %%%%%%%%%% Behavior Labels Figure %%%%%%%%%%%%%%%

    h1 = subplot(number_plots,1,beh_label_plot);
    p = get(h1, 'position');
    p(2) = p(2) + 0.05;
    set(h1, 'position', p );
    y = [1 2];
    hIM2 = imagesc(Total_feats, y, Total_feats, [1 max_feats+length(groups)]);
    axis off
    axis equal tight
    axis([0 Total_feats(end)+1 1 2])
    for i=1:length(Total_feats)
        behav = Total_feats(i);
        y_axis = 1.5;
        text(behav,y_axis,num2str(behav));
    end
    title('Feature Labels')

    cmap = colormap(h1);
    first_color = cmap(1,:);
    % color_interval = ceil((size(cmap,1) - 2)/(length(Total_feats)-1));
    color_interval = floor((size(cmap,1) - 2)/(max_feats+length(groups)-1));
    my_color_map = [];
    my_color_map = [my_color_map; first_color];
    cmap_id = 1;
    % for c=1:length(Total_feats)-2
    for c=1:(max_feats+length(groups))-2
        cmap_id = cmap_id + color_interval;
        my_color_map = [my_color_map; cmap(cmap_id,:)];
    end
    last_color = cmap(end,:);
    my_color_map = [my_color_map; last_color];


    % %%%%%%%%%% Emission Paramaters Figure %%%%%%%%%%%%
    h2 = subplot(number_plots,1,emission_plot);
    p = get(h2, 'position');
    p(2) = p(2) - plot_gap;
    p(4) = p(4) + 0.15;
    set(h2, 'position', p );
    plotEmissionParams(Psi, my_color_map);
%     title('Emission Params')


    %# make all text in the figure to size 14 and bold
    figureHandle = gcf;
    set(findall(figureHandle,'type','text'),'fontSize',17,'fontWeight','bold')  
else
    figure('Units', 'normalized','Color', [1 1 1], 'Position',[0.7107 0.4542  0.1445 0.4567])    
    y = [1 2];
    hIM2 = imagesc(Total_feats, y, Total_feats, [1 max_feats+length(groups)]);
    axis off
    axis equal tight
    axis([0 Total_feats(end)+1 1 2])
    for i=1:length(Total_feats)
        behav = Total_feats(i);
        y_axis = 1.5;
        text(behav,y_axis,num2str(behav));
    end
    title('Feature Labels')
    
end

end