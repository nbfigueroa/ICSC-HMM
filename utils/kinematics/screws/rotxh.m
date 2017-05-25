function R = rotxh(phi)
%ROTXH  return homogeneous rotation matrix around X by PHI
%
%	R = ROTXH(PHI)
%
% See also: ROTY, ROTZ, ROT, POS.

% $ID$
% Copyright (C) 2005, by Brad Kratochvil

R = [1        0         0  0 ; ...
     0 cos(phi) -sin(phi)  0 ; ...
     0 sin(phi)  cos(phi)  0 ; ...
     0        0         0  1 ];

R = roundn(R, -15);

end