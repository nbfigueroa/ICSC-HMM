function [Segmentation Total_feats hIM] = plotSegDataNadia_old( data, Psi, iis )

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
figure('Color',[1 1 1]);
Segmentation = [];

for j=1:length(iis)
    subplot(length(iis),1,j)    
    ii = iis(j); 
    T = data.Ts(ii);
    M = 10;

    X = data.seq(ii);
    Zest = Psi.stateSeq(1,ii).z;
    [seg_results] = findSegPoints(Zest);
end

for j=1:length(iis)
    subplot(length(iis),1,j)    
    ii = iis(j); 
    T = data.Ts(ii);
    M = 10;

    X = data.seq(ii);
    Zest = Psi.stateSeq(1,ii).z;
    
    xs = 1:T;
    %ys = linspace( -3, 3, M);
    ys = linspace(min(min(X)), max(max(X)),M);
    hold all;
%     unique(Zest)
    % hIM = imagesc( xs, ys, repmat(Zest, M, 1), [1 max(Zest)] );
    hIM = imagesc( xs, ys, repmat(Zest, M, 1), [1 max_feats] );
    set( hIM, 'AlphaData', 0.8 );
%     set( hIM, 'AlphaData', 0.7 );

    for i=1:size(X,1)
        if j==1
            c(i,:) = [rand rand rand];
        end
        plot( xs, X(i,:), 'Color',c(i,:), 'LineWidth',2);
    end

    %Draw line at seg points
    [seg_results] = findSegPoints(Zest);
    Segmentation{ii,1} = seg_results;
    yL = get(gca,'YLim');
%     for jj=1:length(seg_results)
    for jj=1:size(seg_results,1)
            model = seg_results(jj,1);       
            seg_end = seg_results(jj,2);
            line([seg_end seg_end],yL,'Color','k','LineWidth',4);
            if jj==1
                start=0;
            else
                start = seg_results(jj-1,2);
            end
%             text(start+(seg_end-start)/2, max(max(X))+10, num2str(model), 'HorizontalAlignment','center','EdgeColor','red','LineWidth',1.5);
%             text(start+(seg_end-start)/2, max(max(X))+10, num2str(model), 'HorizontalAlignment','center');
    end
    figureHandle = gcf;
    %# make all text in the figure to size 14 and bold
    set(findall(figureHandle,'type','text'),'fontSize',20,'fontWeight','bold')
    
%     title( ['Sequence ' num2str(ii)], 'FontSize', 20 );
    axis( [1 T ys(1) ys(end)] );
   

end
%  legend('x','y','z','qw','qi','qj','qk','fx','fy','fz','tx','ty','tz')
%  legend('x','y','z','roll','pitch','yaw','fx','fy','fz','tx','ty','tz')
end