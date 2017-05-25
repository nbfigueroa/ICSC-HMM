function [] = dout(level, varargin)
%DOUT  displays the string if DebugLevel > level 
%
%	DOUT(level, varargin)
%
% DebugLevel is a global and is generally between 0:10. varargin can be any
% string which is normally passed to fprintf
%
% See also: DIMG.

% $Id: dout.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

global DebugLevel; % can be 0-10

if DebugLevel > level,
  fprintf(varargin{:});
end