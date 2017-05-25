function [] = drawskewtraj(omega, theta)
%DRAWSKEWTRAJ  generates a graphical description of a screw over a series
%              of thetas
%
%	DRAWSKEWTRAJ(OMEGA, THETA)
%
% OMEGA is the skew; THETA is a row vector of the different values of theta
% in the trajectory.
%
% See also: DRAWJACOB, DRAWTWIST, DRAWTWISTTRAJ.

% $Id: drawskewtraj.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~isequal([3 1], size(omega)),    
    if ~isskew(omega),
      error('SCREWS:drawskewtraj', 'omega is not a skew');
    end
    % turn it back into a vector
    omega = skewcoords(omega);
  end
  
  hchek = ishold;
  hold on
  
  t = skewexp(omega, theta);

  n = size(t, 3);  
  
  % initialize
  X = zeros(n,4);
  Y = zeros(n,4);
  Z = zeros(n,4);
  
  T = eye(4);
  for j=1:n,
    T(1:3, 1:3) = t(:,:,j);
    
    X = [X;(T * [1;0;0;1])']; % for the x axis
    Y = [Y;(T * [0;1;0;1])'];
    Z = [Z;(T * [0;0;1;1])'];    
  end

  line(X(:,1),X(:,2),X(:,3), 'LineStyle', '-.', 'color', [0.5 0 0])
  line(Y(:,1),Y(:,2),Y(:,3), 'LineStyle', '-.', 'color', [0 0.5 0])
  line(Z(:,1),Z(:,2),Z(:,3), 'LineStyle', '-.', 'color', [0 0 0.5])
 
  % final point
  arrow3([0;0;0], t(1:3,1,n), [1 0 0]);
  arrow3([0;0;0], t(1:3,2,n), [0 1 0]);
  arrow3([0;0;0], t(1:3,3,n), [0 0 1]);      
 
  if hchek == 0
     hold off
  end
  
end