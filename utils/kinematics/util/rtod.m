function [out] = rtod(val)
%RTOD  converts radians to degrees
%
%	OUT = RTOD(val)
%
% See also: DTOR.

% $Id: rtod.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

 out = val/pi*180;