function [] = plotSegBottleCapDataMan(Xn, bestGauPsi, exp, seg)
Xtest = Xn{exp,1};
name = strcat('Segmented Bottle Cap Opening Trial ', num2str(exp));
plotBottlecapData(Xtest(1,:), Xtest(2:7,:)', Xtest(8:30,:)', Xtest(31,:)', name);

    M = 10;
    Zest = bestGauPsi.stateSeq(1,exp).z;

    xs = 1:length(Xtest);

    for i=1:5
        subplot(5,1,i); 
        ys = linspace(min(min(Xtest)), max(max(Xtest)),M);
        hold all;
        hIM = imagesc( xs, ys, repmat(Zest, M, 1), [1 max(Zest)] );
        set( hIM, 'AlphaData', 0.35 );
        yL = get(gca,'YLim');
        for ii=1:length(seg)
            seglims = seg(ii,:);
            seg_in  = seglims(1);
            seg_end = seglims(2);
            if i==1
                c(ii,:) = [0 0 0];
            end
            line([seg_in seg_in],yL,'Color',c(ii,:),'LineWidth',2);
            line([seg_end  seg_end],yL,'Color',c(ii,:),'LineWidth',2);
        end
    end

end