function [] = plotDataNadia( data, ii )

T = data.Ts(ii);
M = 10;

X = data.seq(ii);

xs = 1:T;
ys = linspace(min(min(X)), max(max(X)),M);
hold all;


for i=1:size(X,1)
plot( xs, X(i,:), 'Color',[rand rand rand],'LineWidth',2);
grid on
end

% title( ['Sequence ' num2str(ii)], 'FontSize', 20 );
% title( data.seqNames{ii});

axis( [1 T ys(1) ys(end)] );

end