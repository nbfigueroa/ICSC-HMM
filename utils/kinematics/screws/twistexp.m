function g = twistexp(xi, theta)
%TWISTEXP  calculate the exponential of a twist matrix.
%
%	G = TWISTEXP(S, THETA)
%
% If theta is not specified, we pull it out of xi.  If the first argument to
% twistexp is a vector, twistexp first converts it to a skew-symmetric matrix
% and then takes its exponential.
%
% See also: TWIST, SKEWEXP, TWISTLOG, SKEWLOG.

% $Id: twistexp.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  global DebugLevel;

  if isequal([6 1], size(xi)),
    v = xi(1:3);
    w = xi(4:6);
    w_hat = rot(twist(xi));    
  elseif istwist(xi),
    v = xi(1:3, 4);
    w = skewcoords(xi(1:3,1:3));
    w_hat = xi(1:3,1:3);      
  else
    error('SCREWS:twistexp','xi must be a twist!')
  end

  if 1 == nargin,
    theta = norm(w);
    if (0~=theta),
      w = w/theta;
      w_hat = w_hat/theta;
      v = v/theta;
    else
      theta = 1;
    end
  end
  
  e = eye(3);
  z = [0; 0; 0];
  
  % if it's symbolic, we need to switch the type so isequal works
  if isa(w, 'sym'),
    e = sym(e);
    z = sym(z);
  end  
  
  % loop over multiple thetas
  for i=1:size(theta,2)
  

%     if (isempty(DebugLevel)) || (DebugLevel > 0)  %only use if no debugging  

      if isequal(w, z), % if it is a pure translation
        g(1:3,4,i) = v * theta(i);
        g(1:3,1:3,i) = e;
      else % if it is a rotation and translation
        e = skewexp(w, theta(i));
        g(1:3,1:3,i) = e;
        g(1:3,4,i) = (eye(3) - e)*(w_hat * v) + w*w'*v*theta(i);
      end

      g(4,1:3,i) = 0;
      g(4,4,i) = 1;   
%     else
%       g(:,:,i) = expm(twist(xi)*theta(i));
%     end
   
  end
  

end