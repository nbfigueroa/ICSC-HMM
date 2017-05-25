function [] = drawframetraj(t, scale, interval)
%DRAWFRAMETRAJ  plots the a series of homogeneous transforms
%
%	DRAWFRAMETRAJ(T)
%	DRAWFRAMETRAJ(T, SCALE)
%
% SCALE can be used to determine the size of the coordinate axes.  It
% defaults to 1.
%
% See also: DRAWFRAME, DRAWFRAMETRAJ, ANIMATEFRAMETRAJ.

% $Id: drawframetraj.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil
  
  if 1 == nargin,
    scale = 1;
  end
  
  hchek = ishold;
  hold on  

  n = size(t, 3);  
%   n = size(t, 3)/interval;
  
  % initialize
  X = zeros(n,4);
  Y = zeros(n,4);
  Z = zeros(n,4);
  
  % draw the path
  for j=1:interval:n,
    T = t(:,:,j);  
    if j>1 && j<n
        j = j - interval;
    end
        
    
    X(j,:) = (T * [scale;0;0;1])'; % for the x axis
    Y(j,:) = (T * [0;scale;0;1])'; % for the y axis
    Z(j,:) = (T * [0;0;scale;1])'; % for the z axis
    
    % final axes
    arrow3(pos(T), X(j,1:3)'-pos(T), [0.5 0 0]);
    arrow3(pos(T), Y(j,1:3)'-pos(T), [0 0.5 0]);
    arrow3(pos(T), Z(j,1:3)'-pos(T), [0 0 0.5]);
    
%     if n < 10 || 0 == mod(j,10),
%       text(T(1,4)+0.2*scale, T(2,4)+0.2*scale, T(3,4)+0.2*scale, ...
%            sprintf('%i', j));
%     end
  end
  
%   line(X(:,1),X(:,2),X(:,3), 'LineStyle', '--', 'color', [0.5 0 0])
%   line(Y(:,1),Y(:,2),Y(:,3), 'LineStyle', '--', 'color', [0 0.5 0])
%   line(Z(:,1),Z(:,2),Z(:,3), 'LineStyle', '--', 'color', [0 0 0.5])  
%     
%  
%   % final axes
%   arrow3(pos(T), X(n,1:3)'-pos(T), [1 0 0]);
%   arrow3(pos(T), Y(n,1:3)'-pos(T), [0 1 0]);
%   arrow3(pos(T), Z(n,1:3)'-pos(T), [0 0 1]);      
 
  if 0 == hchek
     hold off
  end
  
  nice3d();
  
end