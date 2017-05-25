function g = fast_twistexp(xi)
%TWISTEXP  Calculate the exponential of a twist matrix.
%
%	G = TWISTEXP(xi)
%
% If theta is not specified, it defaults to 1.  If the first argument to
% twistexp is a vector, twistexp first converts it to a skew-symmetric matrix
% and then takes its exponential.
%
% See also: TWIST, SKEWEXP, SKEWLOG, TWISTLOG.

% $Id: fast_twistexp.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

if ~isequal(size(xi), [6 1]),
  error('SCREWS:fast_twistexp','xi must be in twist coords (6x1)!')
end

theta = norm(xi(4:6));
  
if 0~=theta,
  v = xi(1:3)/theta;
  w = xi(4:6)/theta;
  w_hat = [    0 -w(3)  w(2); ...
            w(3)     0 -w(1); ...
           -w(2)  w(1)     0];
  
  a = w_hat * sin(theta) + w_hat^2 * (1 - cos(theta));
  g = [ eye(3) + a,  -a*w_hat*v + w*w'*v*theta; 0 0 0 1];
else
  g = [ eye(3) xi(1:3); 0 0 0 1];
end

end