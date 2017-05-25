function [Segmentation Total_feats hIM] = plotSegDataNadiaResults( data, iis, Segm_results)

max_feats = max(unique(Segm_results{1}(:,1)));

Total_feats = max_feats;
fprintf('Total Features: %d\n', Total_feats);
figure('Color',[1 1 1]);
Segmentation = [];

for j=1:length(iis)
    subplot(length(iis),1,j)    
    ii = iis(j); 
    T = data.Ts(ii);
    M = 10;

    X = data.seq(ii);
    
    Z_est=zeros(1,Segm_results{ii}(end,2));
    seg_idx = Segm_results{ii}(:,1);
    seg_pts = Segm_results{ii}(:,2);
    for kk=1:length(Segm_results{ii}(:,2))
        seg_end = seg_pts(kk);
        if kk==1
            seg_start = 1;
        else
            seg_start = seg_pts(kk-1);            
        end
        Z_est(seg_start:seg_end) = repmat(seg_idx(kk),1,seg_end-seg_start+1);
    end
%     Zest = Psi.stateSeq(1,ii).z;
%     [seg_results] = findSegPoints(Zest)
end

for j=1:length(iis)
    subplot(length(iis),1,j)    
    ii = iis(j); 
    T = data.Ts(ii);
    M = 10;

    X = data.seq(ii);
%     Zest = Psi.stateSeq(1,ii).z;
    
    Z_est=zeros(1,Segm_results{ii}(end,2));
    seg_idx = Segm_results{ii}(:,1);
    seg_pts = Segm_results{ii}(:,2);
    for kk=1:length(Segm_results{ii}(:,2))
        seg_end = seg_pts(kk);
        if kk==1
            seg_start = 1;
        else
            seg_start = seg_pts(kk-1);            
        end
        Z_est(seg_start:seg_end) = repmat(seg_idx(kk),1,seg_end-seg_start+1);
    end
    
    
    xs = 1:T;
    %ys = linspace( -3, 3, M);
    ys = linspace(min(min(X)), max(max(X)),M);
    hold all;
%     unique(Zest)
    % hIM = imagesc( xs, ys, repmat(Zest, M, 1), [1 max(Zest)] );
    hIM = imagesc( xs, ys, repmat(Z_est, M, 1), [1 max_feats] );
    set( hIM, 'AlphaData', 0.8 );
%     set( hIM, 'AlphaData', 0.7 );

    for i=1:size(X,1)
        if j==1
            c(i,:) = [rand rand rand];
        end
        plot( xs, X(i,:), 'Color',c(i,:), 'LineWidth',2);
    end

    %Draw line at seg points
%     [seg_results] = findSegPoints(Zest)
    seg_results = Segm_results{ii};
    Segmentation{ii,1} = seg_results;
    yL = get(gca,'YLim');
    for jj=1:length(seg_results)
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