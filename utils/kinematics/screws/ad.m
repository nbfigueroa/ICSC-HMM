function out = ad(h)
%AD  Performs the adjoint transform
%
%	A = AD(x)
%
% Computes the adjoint matrix w.r.t X.
%
% See also: IAD, SKEWEXP, TWISTEXP.

% $Id: ad.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if isnumeric(h)
   out = zeros(6,6,size(h,3));
  end

  for i=1:size(h,3),

    if ~ishom(h(:,:,i)),
      error('SCREWS:ad', 'h must be a homogeneous transform') 
    end

    r = rot(h(:,:,i));
    p = pos(h(:,:,i));

    out(1:3,1:3,i) = r;
    out(1:3,4:6,i) = skew(p)*r;
    out(4:6,1:3,i) = 0;  
    out(4:6,4:6,i) = r; 
    
  end
  
end

