function out = noisetwist(xi, np, nt)
%NOISETWIST  applies uniformly distributed noise to a twist
%
%	OUT = NOISETWIST(XI)
%	OUT = NOISETWIST(XI, NP)
%	OUT = NOISETWIST(XI, NP, NT)
%
% XI is the twist; NP is the magnitude of noise to the rotational 
% component of XI; NT is the magnitude of noise to the translational 
% compoenent of XI.
%
% See also: NOISESKEW, NOISEHOM, RANDSKEW.

% $Id: noisetwist.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  switch nargin
    case 0
      out = randtwist();
      return;
    case 1
      np = 0.1;     
      nt = 0.1;
    case 2
      nt = np;
  end
  
  if ~istwist(twist(xi)),
    error('SCREWS:twist', 'xi must be a twist');
  end
  
  z = zeros(3,1);
  
  % we seperate these to keep w/ level 2 calibration errors
  if isequalf(z, xi(4:6,1)), % pure translation
     out(1:3,1) = xi(1:3) + rand(3,1)*np;
     out(4:6,1) = z(3,1);
  elseif isequalf(0, xi(4:6,1)'*xi(1:3,1)), % pure rotation
     out(4:6,1) = noiseskew(xi(4:6), nt);    
     % let's new a new point q to keep this revolute
     [q flag] = lsqr(skew(out(4:6,1)),xi(1:3,1));
     out(1:3, 1) = cross(out(4:6,1),q);
  else
     out(1:3,1) = xi(1:3) + rand(3,1)*np;    
     out(4:6,1) = noiseskew(xi(4:6), nt);
  end
  
  if ~isequal(zeros(6,1), isnan(out)) || ~isequal(zeros(6,1), isinf(out)),
    warning('SCREWS:twist', 'problem applying noise, trying again');
    % recursively try again
    out = noisetwist(xi,np,nt);
  end
  
  if ~istwist(twist(out)),
    error('SCREWS:twist', 'error applying noise');
  end
  
end