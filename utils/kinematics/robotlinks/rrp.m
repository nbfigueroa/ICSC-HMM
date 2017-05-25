function [rn] = rrp()
%RRP  generate a simple roll, roll, prismatic robot
%
%	ROBOT = RRP()
%
% See also: ROBOT.

% $Id: rrp.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  l_x = 0.03; % 30mm
  l_y = 0.03; % 30mm

  l1 = createtwist([0;0;1], [0;0;0]);
  l2 = createtwist([0;-1;0], [0;0;0]);
  l3 = createtwist([0;0;0], [1;0;0]);
  G  = transl(l_x,l_y,0);
  rn = robot({l1;l2;l3},G);
  rn.name = 'rrp';

end