function t = randhom(scale)
%RANDHOM  generates a pseudo-random homogeneous transform
%
%	XI = RANDHOM()
% XI = RANDHOM(SCALE)
%
% SCALE is the magnitude of the translational component
%
% See also: RANDSKEW.

% $Id: randhom.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if 0==nargin,
    scale = 5;
  end
  
  t = eye(4);
  t(1:3,1:3) = skewexp(randskew());
  
  t(1:3,4) = scale * rand(3,1);
  
  if ~ishom(t),
    error('SCREWS:randhom', 'error creating random homogeneous transform');
  end
  
end