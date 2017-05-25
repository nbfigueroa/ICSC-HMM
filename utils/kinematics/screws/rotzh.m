function R = rotzh(alpha)
%ROTX  return homogeneous rotation matrix around Z by ALPHA
%
%	R = ROTZ(ALPHA)
%
% See also: ROTX, ROTY, ROT, POS.

% $ID$
% Copyright (C) 2005, by Brad Kratochvil

R = [cos(alpha) -sin(alpha) 0 0; ...
     sin(alpha)  cos(alpha) 0 0; ...
              0           0 1 0; ...
              0           0 0 1];

R = roundn(R, -15);

end