function [xi_hat] = twistlog(h)
%TWISTLOG  calculate the log of a homogeneous transformation
%
%	[XI_HAT] = TWISTLOG(H)
%
% See also: TWIST, SKEWEXP, TWISTEXP, SKEWLOG.

% $Id: twistlog.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

% this algorithm from Murray pg. 414

global DebugLevel;

if isempty(DebugLevel) || (DebugLevel > 1)
  if ~isrot(R),
    error('ROBOTLINKS:skewlog','H is not a homogeneous transformation')
  end
end

w_hat = skewlog(h(1:3, 1:3));
w = skewcoords(w_hat);

w_norm = norm(w);
p = h(1:3, 4);

if isequalf(zeros(3,1), p),
  A_inv = zeros(3,3);
elseif 0 == sin(w_norm) || 0 == w_norm,
  A_inv = eye(3);
else
  A_inv = eye(3) - ...
          w_hat/2 + ...
          (2*sin(w_norm)-w_norm*(1+cos(w_norm)))/(2*w_norm^2*sin(w_norm))*w_hat*w_hat;
end

xi_hat = [w_hat A_inv*p; 0 0 0 0];
theta = w_norm;