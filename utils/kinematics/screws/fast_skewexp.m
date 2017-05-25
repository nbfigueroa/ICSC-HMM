function g = fast_skewexp(s)
%SKEWEXP  Calculate the exponential of a skew-symmetric matrix.
%
%	G = SKEWEXP(S, THETA)
%
% If theta is not specified, it defaults to 1.  If the first argument to
% skewexp is a vector, skewexp first converts it to a skew-symmetric matrix
% and then takes its exponential.
%
% See also: SKEW, SKEWCOORDS, ROTAXIS.

% $Id: fast_skewexp.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil


  if ~isequal(size(s), [3 1]),
    error('SCREWS:fast_skewexp','r must be in skew coords (3x1)!')
  end

  theta = norm(s);
  
  if 0~=theta,  
    s = skew(s)/theta;
    g = eye(3) + s * sin(theta) + s^2 * (1 - cos(theta));    
  else    
    g = eye(3);
  end

end