function [angles1, angles2] = pairEigenShape(eig1,V1,eig2,V2,color)


pointABC = diag(eig1)^1/2;
pointA = pointABC(1,:);
pointB = pointABC(2,:);
pointC = pointABC(3,:);

% pointA = (V1(:,1).*[sqrt(eig1(1)) 0 0]')'
% pointB = (V1(:,2).*[0 sqrt(eig1(2)) 0]')'
% pointC = (V1(:,3).*[0 0 sqrt(eig1(3))]')'
points = V1*[pointA' pointB' pointC']; % using the data given in the question
xi = pointA-pointB;
xj = pointA-pointC;
theta_x = acosd((xi*xj')/(norm(xi)*norm(xj)));
xi = pointB-pointA;
xj = pointB-pointC;
theta_y = acosd((xi*xj')/(norm(xi)*norm(xj)));
xi = pointC-pointA;
xj = pointC-pointB;
theta_z = acosd((xi*xj')/(norm(xi)*norm(xj)));
angles1 = [theta_x;theta_y;theta_z];

figure('Color',[1 1 1])
fill3(points(1,:),points(2,:),points(3,:),color)
hold on

pointABC = diag(eig2)^1/2;
pointA = pointABC(1,:);
pointB = pointABC(2,:);
pointC = pointABC(3,:);
% 
% pointA = (V2(:,1).*[sqrt(eig2(1)) 0 0]')'
% pointB = (V2(:,2).*[0 sqrt(eig2(2)) 0]')'
% pointC = (V2(:,3).*[0 0 sqrt(eig2(3))]')'
xi = pointA-pointB;
xj = pointA-pointC;
theta_x = acosd((xi*xj')/(norm(xi)*norm(xj)));
xi = pointB-pointA;
xj = pointB-pointC;
theta_y = acosd((xi*xj')/(norm(xi)*norm(xj)));
xi = pointC-pointA;
xj = pointC-pointB;
theta_z = acosd((xi*xj')/(norm(xi)*norm(xj)));
angles2 = [theta_x;theta_y;theta_z];
points=V2*[pointA' pointB' pointC']; % using the data given in the question
fill3(points(1,:),points(2,:),points(3,:),color)
grid on
xlabel('x');ylabel('y');zlabel('z');
axis equal
end