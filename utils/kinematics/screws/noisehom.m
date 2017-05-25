function t = noisehom(T, nr, np)
%NOISEHOM  applies uniformly distributed noise to a homogeneous transform
%
%	T = NOISEHOME(T)
%	T = NOISEHOME(T, NR)
%	T = NOISEHOME(T, NR, NP)
%
% T is the transform; NR is the magnitude of noise to the rotational 
% component of T; NP is the magnitude noise to the translational compoenent 
% of T.
%
% See also: NOISESKEW, NOISETWIST, RANDSKEW.

% $Id: noisehom.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  switch nargin
    case 0
      t = randhom();
      return;
    case 1
      nr = 0.1;     
      np = 0.1;
    case 2
      np = nr;
  end
  
  if ~ishom(T),
    error('SCREWS:noisehom', 'T must be a homogeneous matrix');
  end
  
  t = eye(4);

  t(1:3,1:3) = T(1:3,1:3)*rotx(nr*rand)*roty(nr*rand)*rotz(nr*rand);
  
  t(1:3,4) = T(1:3,4) + np * rand(3,1);
  
  if ~ishom(t),
    error('SCREWS:noisehom', 'error applying noise');
  end
  
end