function w = skewcoords(R)
%SKEWCOORDS  generates a vector w given a skew-symmetric matrix R
%
%	w = SKEWCOORDS(R)
%
% See also: ROTAXIS, SKEWEXP, SKEW.

% $Id: skewcoords.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~isskew(R),
    error('SCREWS:skewcoords','R must be a 3x3 skew-symmetric matrix')
  end
 
  w = [R(3,2);R(1,3);R(2,1)];

end