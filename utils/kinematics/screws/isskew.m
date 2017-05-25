function [out, varargout] = isskew(r)
%ISSKEW  returns true if the matrix is a skew-semmetric matrix
%
%	T = ISSKEW(R)
%	[T REASON] = ISSKEW(R)
%
% if REASON is supplied as an output, a text based message as to why the
% test failed will be returned.
%
% TODO: we should probably check the magnitude of omega  
%
% See also: ISTWIST, ISROT, ISHOM.

% $Id: isskew.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  global DebugLevel;
  
  out = true;
  varargout = {'skew'};
  
  if isempty(DebugLevel) || (DebugLevel > 1)
  
    if ~isequalf(r, -r', 1e-5 ),
      out = false;
      varargout = {'matrix not skew'};
      return;
    end
    
  end
  
end