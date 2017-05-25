function x = homdiff(ta, tn, method)
%HOMDIFF  Compute differential between two homogeneous transforms in twist
%         coordinates
%
%	X = HOMDIFFDIFF(TA, TN)
%
% TA is the actual (measured) frame and TN is the nominal (modeled) frame.
%
% See also: DRAWHOMDIFF, HOMERROR.

% $Id: homdiff.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if 2 == nargin,
    method = 2;
  end

  if ~ishom(ta(:,:,1)) || ~ishom(tn(:,:,1)),
    error('SCREWS:homdiff', 'input not homogeneous transform')
  end

  n = size(ta,3);

  switch method,
    case 1 % taylor series approximation
      for i=1:n,
        x(:,:,i) = twistlog(ta(:,:,i)*inv(tn(:,:,i)));
      end
    case 2  % (ta-tn)*inv(tn)
      for i=1:n,
        x(:,:,i) = ta(:,:,i)*inv(tn(:,:,i))-eye(4);
      end
  end

end
