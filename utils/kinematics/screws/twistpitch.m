function out = twistpitch(xi)
%TWISTPITCH  inputs a twist and returns the pitch
%
%	OUT = TWISTPITCH(XI)
%
% This function returns the pitch of a twist
%
% See also: TWISTMAGNITUDE, TWISTAXIS.

% $Id: twistpitch.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if isequal([6 1], size(xi)),
    xi = twist(xi);
  end

  if ~istwist(xi),
    error('SCREWS:twistpitch','xi must be a twist!')
  end

  omega = skewcoords(rot(xi));
  v = pos(xi);  
  
  if zeros(3,1) == omega,
    out = inf;
  else
    out = omega'*v/norm(omega)^2;
  end  

end