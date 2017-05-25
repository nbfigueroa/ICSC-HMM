function axis = rotaxis(R, theta)
%ROTAXIS  calculate the axis of rotation for a matrix R
%
%	OMEGA = ROTAXIS(R, THETA)
%
% Given a rotation matrix R and a known THETA, this function
% returns an axis OMEGA .  THETA is limited to a range between
% 0 and pi.  There can potentially be multiple OMEGA values 
% that are correct, so this function will return an arbitrary
% one of them.
%
% See also: ROTPARAM, SKEWEXP.

% $Id: rotaxis.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~isrot(R),
    error('SCREWS:rotaxis','R must be a rotation matrix')
  end

  if theta < 0 || theta > pi,
    error('SCREWS:rotaxis','theta must be between 0 and pi');
  end
  
  if isequalf(pi,theta) || isequalf(0,theta),
    axis = null(R-eye(3),'r');
    axis = axis(:,1)/norm(axis(:,1));
  else
    axis = [R(3,2) - R(2,3); R(1,3) - R(3,1); R(2,1) - R(1,2) ];
    axis = axis/(2*sin(theta));
  end
  
end
