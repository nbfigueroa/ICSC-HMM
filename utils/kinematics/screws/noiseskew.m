function out = noiseskew(omega, n)
%NOISESKEW  applies uniformly distributed noise to a skew
%
%	OUT = NOISESKEW(OMEGA)
%	OUT = NOISESKEW(OMEGA, N)
%	OUT = NOISESKEW(OMEGA, N)
%
% OMEGA is the skew; NP is the magnitude of noise
%
% See also: NOISETWIST, RANDSKEW.

% $Id: noiseskew.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  switch nargin
    case 0
      out = randskew();
      return;
    case 1
      n = 0.1;     
  end
  
  if ~isskew(skew(omega)),
    error('SCREWS:noiseskew', 'error creating skew');
  end  
  
  out = omega + rand(3,1)*n;
  out = out/norm(out);
  
  if ~isskew(skew(out)),
    error('SCREWS:noiseskew', 'error applying noise');
  end    
  
end