function g = skewexp(s, theta)
%SKEWEXP  calculate the exponential of a skew-symmetric matrix
%
%	G = SKEWEXP(S, THETA)
%
% If theta is not specified, it defaults to 1.  If the first argument to
% skewexp is a vector, skewexp first converts it to a skew-symmetric matrix
% and then takes its exponential.
%
% See also: SKEW, SKEWCOORDS, ROTAXIS.

% $Id: skewexp.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if 1 == nargin,
    theta = 1;
  end
  
  if isequal([3 1], size(s)),
    s = skew(s);
  end

  if ~isskew(s),
    error('SCREWS:skewexp','s must be a 3x3 skew-symmetric matrix')
  end

   
  for i=1:size(theta,2),
    g(:,:,i) = eye(3) + s * sin(theta(i)) + s^2 * (1 - cos(theta(i)));
  end
    

end