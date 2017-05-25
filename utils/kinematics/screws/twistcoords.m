function t = twistcoords(xi)
%TWISTCOORDS  convert xi from a 4 x 4 skew symmetric matrix to a 6-vector
%
%	t = TWISTCOORDS(XI)
%
% The format of t is [v1; v2; v3; w1; w2; w3]
%
% See also: TWIST, HOMTOTWIST, TWISTEXP.

% $Id: twistcoords.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~istwist(xi),
    error('SCREWS:twistcoords','xi must be a twist matrix!')
  end
  
  t = [xi(1:3, 4); skewcoords(xi(1:3, 1:3))];
  
end