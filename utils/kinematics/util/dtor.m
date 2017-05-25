function [out] = dtor(val)
%DTOR  converts degrees to radians
%
%	OUT = DTOR(val)
%
%
% See also: RTOD.

% $Id: dtor.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

 out = val/180*pi;