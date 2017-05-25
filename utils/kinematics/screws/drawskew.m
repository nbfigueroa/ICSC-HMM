function [varargout] = drawskew(omega)
%DRAWSKEW  plot a skew's axis of rotation
%
%	LHNDL = DRAWSKEW(W, Q, H)
%
% See also: DRAWROBOT, DRAWTWIST.

% $Id: drawskew.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~isequal([3 1], size(omega)),
    error('SCREWS:drawskew','omega not a skew');
  end
 
  % draw the axis line
  lhndl = plot3([-omega(1) omega(1)], ...
                [-omega(2) omega(2)], ...
                [-omega(3) omega(3)], '-ok');

  if 1 == nargout,
    varargout = {lhndl};
  end      
      
end