function omega = randskew()
%RANDSKEW  generates a pseudo-random skew vector
%
%	OMEGA = RANDSKEW()
%
% The norm of omega is normalized to 1;
%
% See also: RANDTWIST.

% $Id: randskew.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  omega = 2*rand(3,1)-1;
  omega = omega/norm(omega);
  
  if ~isskew(skew(omega)),
    error('SCREWS:randskew','error creating skew');
  end  
  
end