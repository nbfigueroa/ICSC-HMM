function out = iad(xi)
%IAD  Performs the inverse adjoint transform
%
%	A = IAD(x)
%
% Computes the inverse of the adjoint matrix w.r.t X.
%
% See also: AD, SKEWEXP, TWISTEXP.

% $Id: iad.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $

% Copyright (C) 2005, by Brad Kratochvil

  for i=1:size(xi,3),

    if ~ishom(xi(:,:,i)),
      error('SCREWS:iad','x must be a homogeneous transform') 
    end

    r = rot(xi(:,:,i));
    p = pos(xi(:,:,i));

    out(1:3,1:3,i) = r';  
    out(1:3,4:6,i) = -r'*skew(p);
    out(4:6,1:3,i) = 0;  
    out(4:6,4:6,i) = r';  
    
  end
  
end

