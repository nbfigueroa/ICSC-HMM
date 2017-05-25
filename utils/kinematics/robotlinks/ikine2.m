function [theta] = ikine2(rn, pose, joints, tol)
%IKINE2  inverse kinematics for serial link manipulator
%
%	THETA = IKINE2(ROBOT, POSE, X0, TOL)
%
% Computes the inverse kinematics for robot ROBOT to reach pose POSE.
% ROBOT is a r object.  This method uses an analytic solution where 
% possible, otherwise it uses the Jacobian inverse.
% 
% See also: ikine, fkine.

% $Id: ikine2.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

% default tolerance if one is not provided
if ~exist('tol', 'var'),
  tol = 1e-9;
end

if 4==size(pose,1) && 4==size(pose,1),  
  for i=1:size(pose,3),
    se3(:,i) = twistcoords(twistlog(pose(:,:,i)));
  end
else
  se3 = pose;
end

% use the home position for an initial guess if one is not provided
if ~exist('joints', 'var'),
    joints = zeros(rn.n, size(se3,2));
end


% timeout after 1000 iterations
timeout = 1000;

% variables for the various different methods
beta = 0.5;
d = 0.5;
lambda = 0.1;

    
  for ix=1:size(se3,2),

    est = twistcoords(twistlog(fkine(rn, joints(:,ix))));
    count = 0;
    
    while (norm(se3(:,ix)-est) > tol) && (count < timeout),
      J = sjacob(rn, joints(:,ix));
      
      % pseudo inverse
      joints(:,ix) = joints(:,ix) + beta * pinv(J)*(se3(:,ix)-est);

%       % jacobian transpose
%       joints(:,ix) = joints(:,ix) + beta * J'*(se3(:,ix)-est);

%       % damped least squares
%       err = (se3(:,ix)-est);
%       
%       if (norm(err) < d),
%         
%       else
%         err = d * err/norm(err);
%       end
%       joints(:,ix) = joints(:,ix) + beta * J'*inv(J*J'+lambda*lambda*eye(size(J,1)))*(err);
      
      est = twistcoords(logm(fkine(rn, joints(:,ix))));
      count = count + 1;
    end
    
    if (count >= timeout),
      warning('ROBOTLINKS:robot', 'ikine2 timed out');
    end

    
    for i=1:size(joints(:,ix),1),
      % if it's a revolute joint
      if ~isequal(rn.xi{i}(4:6), zeros(3,1)),
        % normalize to +- pi
        joints(i,ix) = atan2(sin(joints(i,ix)), cos(joints(i,ix)));
        
        if isequalf(joints(i,ix), 2*pi)
          joints(i,ix) = 0;
        end
      end      
    end
    
    theta(:, ix) = joints(:,ix);
    
  end
