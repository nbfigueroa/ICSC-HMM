function [out, varargout] = ishom(t)
%ISHOM  returns true if the matrix is a homogeneous transform
%
%	OUT = ISHOM(T)
%	[OUT REASON] = ISHOM(T)
%
% if REASON is supplied as an output, a text based message as to why the
% test failed will be returned.
%
% See also: ISTWIST, ISROT.

% $Id: ishom.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  global DebugLevel;

  out = true;
  varargout = {'hom'};
  
  if ~isequal([4 4], size(t)),
    out = false;
    varargout = {'matrix not 4x4'};
    return;
  end

  if isempty(DebugLevel) || (DebugLevel > 1)
    if ~isrot(t(1:3,1:3)),
      out = 0;
      varargout = {'R is not a rotation matrix'};
      return;    
    end

    if ~isequalf(1, t(4,4)),
      out = false;
      varargout = {'t(4,4) is not 1'};
      return;    
    end

    if ~isequalf(zeros(1,3), t(4,1:3)),
      out = false;
      varargout = {'t(4,1:3) are not 0'};
      return;    
    end
  end
  
end