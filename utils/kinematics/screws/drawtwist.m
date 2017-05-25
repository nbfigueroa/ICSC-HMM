function [varargout] = drawtwist(xi)
%DRAWTWIST  plot a twist's axis of rotation
%
%	LHNDL = DRAWTWIST(XI)
%
%
% See also: DRAWROBOT, DRAWSKEW.

% $Id: drawtwist.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  varargout = [];

  if ~isequal([6 1], size(xi)),
    error('SCREWS:drawtwist', 'xi not a twist');
  end
 
  v = xi(1:3);
  omega = zeros(3,1);
  if (0 ~= norm(xi(4:6))),
    omega = xi(4:6)/norm(xi(4:6));  %normalize the length
  end
%   axis = twistaxis(xi) + omega;
  axis = twistaxis(xi);

  % draw the axis line
  if isequal(zeros(3,1), omega), % pure translation
    lhndl = plot3([-v(1) v(1)], ...
                  [-v(2) v(2)], ...
                  [-v(3) v(3)], '-sr'); 
  elseif isequal(zeros(3,1), v), % pure rotation
    lhndl = plot3([-omega(1) omega(1)], ...
                  [-omega(2) omega(2)], ...
                  [-omega(3) omega(3)], '-*r');    
  else
    lhndl = plot3([axis(1)-2*omega(1) axis(1)+2*omega(1)], ...
                  [axis(2)-2*omega(2) axis(2)+2*omega(2)], ...
                  [axis(3)-2*omega(3) axis(3)+2*omega(3)], '-*r');
  end
  
  if 1 == nargout,
    varargout = {lhndl};
  end

end