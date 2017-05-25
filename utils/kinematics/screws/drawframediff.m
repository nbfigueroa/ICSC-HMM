function [] = drawframediff(t1, t2)
%DRAWFRAMEDIFF  plots the difference between t1 and t2
%
%	DRAWFRAMEDIFF(T1, T2)
%
% plots the difference between homogeneous transform T1 and T2.
%
% See also: DRAWFRAME, DRAWFRAMETRAJ, ANIMATEFRAMETRAJ.

% $Id: drawframediff.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~ishom(t1(:,:,1)) || ~ishom(t2(:,:,1)),
    error('SCREWS:drawframediff', 't is not a homogeneous transform');
  end
  
  % to clear the plot
  plot(0,0);
  
  hchek = ishold;
  hold on  

  n = size(t1, 3);  
  
  % draw the path 
  for j=1:n,
    XA = (t1(:,:,j) * [1;0;0;1])';
    XN = (t2(:,:,j) * [1;0;0;1])';

    YA = (t1(:,:,j) * [0;1;0;1])';
    YN = (t2(:,:,j) * [0;1;0;1])';

    ZA = (t1(:,:,j) * [0;0;1;1])';
    ZN = (t2(:,:,j) * [0;0;1;1])';
    
    p = pos(t1(:,:,j));
    r = rot(t1(:,:,j));
    
    arrow3(p, r(1:3,1), [0.5 0 0], 0.5);
    arrow3(p, r(1:3,2), [0 0.5 0], 0.5);
    arrow3(p, r(1:3,3), [0 0 0.5], 0.5);
    
    arrow3(XA(1:3)', (XA(1:3)-XN(1:3))', 'r');
    arrow3(YA(1:3)', (YA(1:3)-YN(1:3))', 'g');    
    arrow3(ZA(1:3)', (ZA(1:3)-ZN(1:3))', 'b'); 
  end

  if 0 == hchek
     hold off
  end
  
end