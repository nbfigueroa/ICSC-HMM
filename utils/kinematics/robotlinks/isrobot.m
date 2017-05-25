function [out, varargout] = isrobot(r)
%ISROBOT  Returns 1 if the matrix is a robot
%
%	T = ISROBOT(R)
%	[T REASON] = ISTROBOT(R)
%
% if REASON is supplied as an output, a text based message as to why the
% test failed will be returned.
%
% See also: ISROT, ISSKEW, ISTWIST.

% $Id: isrobot.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  global DebugLevel;

  out = 1;
  varargout = {'robot'};
  
  if ~isa(r, 'struct'),
    out = 0;
    varargout = {'a robot must be a struct'};
    return;
  end
  
  if r.n < 1,
    out = 0;
    varargout = {'a robot must have links!'};
    return;
  end

  if (isempty(DebugLevel)) || (DebugLevel > 1)  
  
    for i=1:r.n,
      if ~istwist(r.xi{i}),
        out = 0;
        varargout = {'link is not a twist!'};
        return;
      end
    end

    if ~ishom(r.g_st),
      out = 0;
      varargout = {'last element is not a homogeneous transform!'};
      return;
    end
  
  end
  
end