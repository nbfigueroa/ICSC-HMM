function [out, varargout] = istwist(xi)
%ISTWIST  returns true if the matrix is a twist
%
%	T = ISTWIST(XI)
%	[T REASON] = ISTWIST(XI)
%
% if REASON is supplied as an output, a text based message as to why the
% test failed will be returned.
%
% TODO: we should probably check the magnitude of omega
%
% See also: ISROT, ISSKEW, ISHOM.

% $Id: istwist.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil
  
  global DebugLevel;
 
  out = true;
  varargout = {'twist'};
  
  if isequal([6 1], size(xi)),
    return;  
  elseif ~isequal([4 4], size(xi)),
    out = false;
    varargout = {'matrix must be 4x4'};
    return;
  end
  
  if isempty(DebugLevel) || DebugLevel > 1  
  
    if ~isequalf(zeros(4,1), diag(xi), 1e-10),
      out = false;
      varargout = {'diagonal elements must be 0'};
      return;
    end
    
  end
  
end