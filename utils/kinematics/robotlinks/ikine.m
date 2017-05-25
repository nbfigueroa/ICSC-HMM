function [theta] = ikine(rn, pose, x0, tol)
%IKINE  inverse kinematics for serial link manipulator
%
%	THETA = IKINE(ROBOT, POSE, X0, TOL)
%
% Computes the inverse kinematics for robot ROBOT to reach pose POSE.
% ROBOT is a r object.  This method uses the lsqnonlin functions available
% in Matlab.
% 
% See also: ikine2, fkine.

% $Id: ikine.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if 4==size(pose,1) && 4==size(pose,1),  
    for i=1:size(pose,3),
      se3(:,i) = twistcoords(twistlog(pose(:,:,i)));
    end
  else
    se3 = pose;
  end

  % use the home position for an initial guess if one is not provided
  if ~exist('x0', 'var'),
    % the rand provides a little nudge
    x0 = zeros(rn.n, size(se3,2)) + 0.1*rand(3,size(se3,2));
  end

  % default tolerance if one is not provided
  if ~exist('tol', 'var'),
    tol = 1e-9;
  end
 
  options = optimset();

%   options = optimset(options, 'Jacobian', 'on');
%   options = optimset(options, 'DerivativeCheck', 'on');
%   options = optimset(options, 'Diagnostics', 'on');
%   options = optimset(options, 'Display', 'iter');
  options = optimset(options, 'TolX', tol);
  options = optimset(options, 'TolFun', tol);
  options = optimset(options, 'MaxFunEvals', 1e5);
  options = optimset(options, 'Display', 'off');

  lb = [];
  ub = [];
  
  for j=1:size(se3,2),

    [theta(:,j) resnorm residual exitflag output] = lsqnonlin(@fun, x0(:,j), lb, ub, options);
        
  end
 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  function [F, J] = fun(x) % find the SSQ difference

    F = twistcoords(twistlog(fkine(rn, x))) - se3(:,j);
           
    % we need to provide the jacobian
    if nargout > 1
      % calculate the current jacobian
      J = sjacob(rn, x);
    end

  end
  
end