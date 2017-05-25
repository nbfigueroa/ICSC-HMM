function out = twistmagnitude(xi)
%TWISTMAGNITUDE  inputs a twist and returns the magnitude
%
%	OUT = TWISTMAGNITUDE(XI)
%
% This function returns the axis of a twist
%
% See also: TWISTAXIS, TWISTPITCH.

% $Id: twistmagnitude.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if isequal([6 1], size(xi)),
    xi = twist(xi);
  end

  if ~istwist(xi),
    error('SCREWS:twistmagnitude','xi must be a twist!')
  end

  omega = skewcoords(rot(xi));
  v = pos(xi);
  
  if zeros(3,1) == omega,
    out = norm(v);
  else
    out = norm(omega);
  end

end