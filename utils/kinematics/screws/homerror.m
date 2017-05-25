function [d_theta d_b] = homerror(ta, tn)
%HOMEERROR  calculates the error between two homogeneous transforms
%
%	[D_THETA D_B] = HOMEERROR(TA, TN)
%
% TA is the actual (measured) frame and TN is the nominal (modeled) frame.
% This function uses the metric suggested by Park and Okamura.
%
% See also: HOMDIFF.

% $Id: homerror.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~ishom(ta(:,:,1)) || ~ishom(tn(:,:,1)),
    error('SCREWS:homerror', 'input not homogeneous transform')
  end

  n = size(ta,3);
  
  d_theta = zeros(n,1);
  d_b = zeros(n,1);
 
  for i=1:n,
    % metric from park & okamura
    theta_a = rot(ta(:,:,i));
    theta_n = rot(tn(:,:,i));
    
    b_a = pos(ta(:,:,i));
    b_n = pos(tn(:,:,i));    
    
    d_theta(i) = norm(twistlog(inv(theta_a)*theta_n));
    d_b(i) = norm(b_a - b_n);
    
%     d_theta(i) = norm(logm(inv(rot(tn(:,:,i)))*rot(ta(:,:,i))));
%     d_b(i) = norm(pos(ta(:,:,i))-pos(tn(:,:,i)));


  end

end
