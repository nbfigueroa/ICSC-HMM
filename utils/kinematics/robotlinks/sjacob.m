function J = sjacob(r, theta)
%SJACOB  calculate the spatial jacobian for the robot
%
%	J = SJACOB(ROBOT, THETA)
%
% Computes the calibration jacobian for ROBOT with joint variables THETA.
%
% See also: ROBOT, CJACOB, BJACOB.

% $Id: sjacob.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~isrobot(r),
    error('ROBOTLINKS:sjacob', 'r is not a robot');
  end
    
  if ~isequal(r.n, size(theta,1)),
    error('ROBOTLINKS:sjacob', 'incorrect number of theta values');
  end
  
  J = zeros(6, r.n, size(theta,2));
  
  for k=1:size(theta, 2),
    
    J(:,1,k) = r.xi{1};
    
    for j=2:r.n,
      p = eye(4);
      for i=1:j-1,
        p = p * twistexp(r.xi{i}, theta(i,k));
%         p = p * expm(twist(r.xi{i}) * theta(i,k));
      end
      J(:,j,k) = ad(p) * r.xi{j};

  %     % an alternate method from Park
  %     P = eye(6);     
  %     for i=1:j-1,
  %       P = P * ad(twistexp(r.xi{i}, theta(i)));
  %     end
  %       J = [J P*r.xi{j}];
    end
    
  end
  
end