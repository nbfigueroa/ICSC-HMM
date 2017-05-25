function R = rot(x)
%ROT extracts the rotational part of a homogeneous matrix
%
%	R = ROT(x)
%
% extracts the rotational part of a matrix
%
% NOTE: We don't check the size of the matrix because we want this function
%       to return as fast as possible.
%
% See also: pos.

% $Id: rot.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  R = x(1:3,1:3);

end