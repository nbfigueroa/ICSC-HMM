function t = twist(xi)
%TWIST  convert xi from a 6-vector to a 4 x 4 skew-symmetric matrix
%
%	t = TWIST(XI)
%
% The format of t is [v1; v2; v3; w1; w2; w3]
%
% See also: TWISTCOORDS, SKEW, RANDTWIST.

% $Id: twist.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if isequal([4 4], size(xi)),
    t = xi;
    return;    
  elseif ~isequal([6 1], size(xi)),
    error('SCREWS:twist','xi must be a 6x1 matrix!')
  end
  
  if ~isa(xi, 'sym'),    
    t = zeros(4);
  end
 
  t(1:3, 1:3) = skew(xi(4:6));
  t(1:3, 4) = xi(1:3);  
  
end