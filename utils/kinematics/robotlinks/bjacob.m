function J = bjacob(r, theta)
%BJACOB  Calculate the body jacobian for the r
%
%	J = BJACOB(ROBOT, THETA)
%
% Computes the calibration jacobian for ROBOT with joint variables THETA.
%
% See also: ROBOT, CJACOB, SJACOB.

% $Id: bjacob.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~isrobot(r),
    error('ROBOTLINKS:sjacob', 'r is not a robot');
  end
    
  if ~isequal(r.n, size(theta,1)),
    error('ROBOTLINKS:sjacob', 'incorrect number of theta values');
  end
  
  J = zeros(6, r.n, size(theta,2));
  
  for k=1:size(theta, 2),
   
    for j=1:r.n,
      p = eye(4);
      for i=j:r.n,
        p = p * expm(twist(r.xi{i}) * theta(i,k));
      end
      
      p = p * r.g_st;
      J(:,j,k) = iad(p) * r.xi{j};

    end
    
  end
  
end