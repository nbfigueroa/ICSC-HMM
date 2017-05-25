function R = rotyh(beta)
%ROTY  return homogeneous rotation matrix around Y by BETA
%
%	R = ROTY(BETA)
%
% See also: ROTX, ROTZ, ROT, POS.

% $ID$
% Copyright (C) 2005, by Brad Kratochvil

R = [cos(beta) 0 sin(beta) 0; ...
             0 1         0 0; ...
    -sin(beta) 0 cos(beta) 0; ...
             0 0         0 1];
  
  
R = roundn(R, -15);


end