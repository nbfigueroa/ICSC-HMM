function [omega theta] = rotparam(r)
%ROTPARAM  pulls a skew matrix and theta out of a rotation matrix
%
%	[OMEGA THETA] = ROTPARAM(R)
%
% Note: This function only returns theta values on the range of 0 to Pi.
%
% See also: ROTAXIS, HOMTOTWIST, SKEWLOG.

% $Id: rotparam.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~isrot(r),
    error('SCREWS:rotparam', 'T must be a rotation');
  end
  
  %trace(r) = sum of eigenvalues
  t = (trace(r)-1)/2;
  
  % this is only in case of numerical errors
  if t < -1,
    t = -1;
  elseif t>1,
    t = 1;
  end
  
  % from Murray pg 29-30
  
  theta = acos(t);
  
  omega = rotaxis(r, theta);
  
end