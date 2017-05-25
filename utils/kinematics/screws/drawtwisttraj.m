function [] = drawtwisttraj(xi, theta)
%DRAWTWISTTRAJ  generates a graphical description of a twist over a series
%               of thetas
%
%	DRAWTWISTTRAJ(XI, THETA)
%
% TWIST is the twist; THETA is a row vector of the different values of theta
% in the trajectory.
%
% See also: DRAWJACOB, DRAWSKEW, DRAWTWISTTRAJ.

% $Id: drawtwisttraj.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~isequal([6 1], size(xi)),
    if ~istwist(xi),
      error('SCREWS:drawtwisttraj', 'xi is not a twist');
    end
    % turn it back into a vector
    xi = twistcoords(xi);
  end
  
  if (1 == nargin) || (0 == size(theta, 1)),
    error('SCREWS:drawtwisttraj', 'no theta values received');
  end

  drawtwist(xi);
  
  hchek = ishold;
  hold on  
  
  t = twistexp(xi, theta);
  n = size(t, 3);  
  
  % initialize
  X = zeros(1,4);
  Y = zeros(1,4);
  Z = zeros(1,4);
  
  % draw the path
  for j=1:n,
    T = t(:,:,j);    
    X = [X;(T * [1;0;0;1])']; % for the x axis
    Y = [Y;(T * [0;1;0;1])']; % for the y axis
    Z = [Z;(T * [0;0;1;1])']; % for the z axis
  end

  line(X(:,1),X(:,2),X(:,3), 'LineStyle', '-.', 'color', [0.5 0 0])
  line(Y(:,1),Y(:,2),Y(:,3), 'LineStyle', '-.', 'color', [0 0.5 0])
  line(Z(:,1),Z(:,2),Z(:,3), 'LineStyle', '-.', 'color', [0 0 0.5])
 
  % final axes
  arrow3(pos(T), X(n+1,1:3)'-pos(T), [1 0 0]);
  arrow3(pos(T), Y(n+1,1:3)'-pos(T), [0 1 0]);
  arrow3(pos(T), Z(n+1,1:3)'-pos(T), [0 0 1]);      
 
  if 0 == hchek
     hold off
  end
  
end