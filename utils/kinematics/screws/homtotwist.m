function [xi theta] = homtotwist(t)
%HOMTOTWIST  finds a twist and a theta to create the homogeneous transform T 
%
%	[XI THETA] = HOMTOTWIST(T)
%
% Note: There are multiple possible combinations of xi and theta that satisfy
%       exp(xi*theta) = T.  Theta is thus returned with a range of 0 to pi.
%
% See also: ROTPARAM, ROTAXIS, TWISTEXP, TWISTLOG, SKEWLOG.

% $Id: homtotwist.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~ishom(t),
    error('SCREWS:homtotwist', 'T must be a homogeneous transform');
  end
  
  % preallocate xi
  xi = zeros(6,1);

  % this algorithm from Murray pg. 43

  % it's only a translation  
  if isequalf(rot(t), eye(3)),
    % if (R,p) == (I, 0)
    if isequalf(pos(t), zeros(3,1)),
      theta = 0;
    else  
      theta = norm(pos(t));
      xi(1:3) = pos(t)/theta;
    end
  else
    r = rot(t);
    [omega theta] = rotparam(r);
    omega_hat = skew(omega);
    if theta == 0
        theta = 0.1;
    end
    A = (eye(3) - skewexp(omega_hat, theta))*omega_hat + omega*omega'*theta;
    v = linsolve(A, pos(t));
    
    xi(1:3) = v;
    xi(4:6) = omega;
  end
  
end
