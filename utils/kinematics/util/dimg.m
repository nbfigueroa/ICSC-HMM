function [] = dimg(level, name, img)
%DIMG  displays an image if DebugLevel > level 
%
%	DIMG(level, name, img)
%
% DebugLevel is a global and is generally between 0:10
%
% See also: DOUT.

% $Id: dimg.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

global DebugLevel;

if DebugLevel > level,
  named_figure(name);
  
  imagesc(img);
  if 1 == size(img,3),
    colormap gray;
  end

end