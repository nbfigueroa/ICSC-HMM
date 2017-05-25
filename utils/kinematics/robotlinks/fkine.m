function t = fkine(r, theta)
%FKINE  forward kinematics for serial link manipulator
%
%	TR = FKINE(ROBOT, THETA)
%
% Computes the forward kinematics for each joint variable theta.
% ROBOT is a r object.
% 
% If THETA is a matrix, the rows are interpretted as the generalized 
% joint coordinates for a sequence of points along a trajectory.  
% THETA(i,j) is the i'th joint parameter for the j'th trajectory point.  
% In this case FKINE(ROBOT, THETA) returns 3D matrix where the last 
% subscript is the index along the path.
%
% See also: TR2DIFF, PLOT_HOM.

% $Id: fkine.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~isrobot(r),
    error('ROBOTLINKS:fkine','r is not a robot')
  end  

  if size(theta,1) ~= r.n,
    error('ROBOTLINKS:fkine','wrong number of rows for theta')
  end
 
  t = zeros(4,4,size(theta,2));

  for j=1:size(theta, 2),
    t(:,:,j) = eye(4);
    for i=1:r.n
      t(:,:,j) = t(:,:,j)*twistexp(r.xi{i}, theta(i,j));
%       t(:,:,j) = t(:,:,j)*expm(twist(r.xi{i}) * theta(i,j));      
%       t(:,:,j) = t(:,:,j)*fast_twistexp(r.xi{i} * theta(i,j));      
    end
    t(:,:,j) = t(:,:,j)*r.g_st;
  end
   
end
