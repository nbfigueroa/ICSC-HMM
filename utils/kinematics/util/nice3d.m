function nice3d()
%NICE3D  make 3d plots nicer
%
%	NICE3D
%
% Make 3d plots nicer by using showing bounding box, making axis units
% equal, showing grid, and allowing rotation by mouse dragging

% $Id: nice3d.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

xlabel('x'); 
ylabel('y'); 
zlabel('z');

box('on');
grid('on');
axis('tight');
axis('equal');
a = axis(gca);
b = 0.05*(a(2:2:end)-a(1:2:end));
a(1:2:end) = a(1:2:end) - b;
a(2:2:end) = a(2:2:end) + b;
axis(a);
axis('vis3d');

% turn rotate3d on
cameratoolbar('SetMode','orbit');


