function [] = plotDataMin(X)
% 
% T = data.Ts(ii);
T = length(X);
M = 5;
% 
% X = data.seq(ii);

xs = 1:T;
%ys = linspace( -3, 3, M);
ys = linspace(min(min(X)), max(max(X)),M);
hold all;
% hIM = imagesc( xs, ys, repmat(data.zTrue(ii), M, 1), [1 max( data.zTrueAll)] );
% set( hIM, 'AlphaData', 0.65 );

plot( xs, X(1,:), 'k.-' );
plot( xs, X(2,:), 'r.-' );
plot( xs, X(3,:), 'm.-' );
plot( xs, X(4,:), 'g.-' );
plot( xs, X(5,:), 'b.-' );
plot( xs, X(6,:), 'c.-' );
plot( xs, X(7,:), 'y.-' );
% plot( xs, X(8,:), 'w.-' );
% title( ['Sequence ' num2str(ii)], 'FontSize', 20 );

axis( [1 T ys(1) ys(end)] );

end