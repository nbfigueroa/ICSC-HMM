function [R] = rotationMatrx(type, angle)
% ----Input---
% type: Basic Rotation types (X,Y,Z)
% angle: Rotation angle in radians
R = eye(3,3);
switch type
    case 'x'
       R(2:3,2:3) = [cos(angle) -sin(angle); sin(angle) cos(angle)];
    case 'y'
        R(1,:) = [cos(angle) 0 sin(angle)];
        R(3,:) = [-sin(angle) 0 cos(angle)];
    case 'z'
        R(1:2,1:2) = [cos(angle) -sin(angle); sin(angle) cos(angle)];
end
end