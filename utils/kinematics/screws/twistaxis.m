function [l1 l2] = twistaxis(xi)
%TWISTAXIS  inputs a twist and returns the axis
%
%	[L1 L2] = TWISTAXIS(XI)
%
% This function returns the axis of a twist
%
% See also: TWISTMAGNITUDE, TWISTPITCH.

% $Id: twistaxis.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if isequal([6 1], size(xi)),
    xi = twist(xi);
  end

  if ~istwist(xi),
    error('SCREWS:twistaxis','xi must be a twist!')
  end

  omega = skewcoords(rot(xi));
  v = pos(xi);
  
  % from Murray pg 48
  if isequalf(zeros(3,1), omega),
    l1 = [0;0;0];
    l2 = v;
  else
    l1 = skew(omega)*v/norm(omega)^2;
    l2 = omega;
  end

end