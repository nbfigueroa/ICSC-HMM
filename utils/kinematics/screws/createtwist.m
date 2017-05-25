function [V] = createtwist(omega, q)
%CREATETWIST  Inputs a skew and a point, and returns a twist
%
%	V = CREATETWIST(OMEGA, Q)
%
% This function makes sure that the OMEGA magnitude is normalized to 1.
%
% See also: TWIST, SKEW, RANDTWIST, RANDSKEW.

% $Id: createtwist.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if isequal(omega, [0; 0; 0]), % if it is a pure translation
    if ~isa(q, 'sym'),
      q = q ./ norm(q);
    end    
    V(1:3,1) = q;
  else
    if ~isa(omega, 'sym'),
      omega = omega ./ norm(omega);
    end
    V(1:3,1) = -cross(omega, q);  
  end

  V(4:6,1) = omega;
  
end