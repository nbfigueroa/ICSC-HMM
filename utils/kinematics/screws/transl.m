function T = transl(x, y, z)
%TRANSL set or extract the translational part of a homogeneous matrix
%
%	T = TRANSL(x)
%	T = TRANSL(x, y, z)
%
% If only X is set, TRANSL extracts the translational part of a matrix.
% Otherwise, it creates a translational matrix.
%
% NOTE: We don't check the size of the matrix because we want this function
%       to return as fast as possible.

% $Id: transl.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if 1 == nargin,
    T = x(1:3,4);
    T(4,1) = 1;
  else
    T = eye(4);
    if isa(x, 'sym') || isa(y, 'sym') || isa(z, 'sym'),
      T = sym(T);
    end  
    T(1,4) = x;
    T(2,4) = y;
    T(3,4) = z;
  end

end