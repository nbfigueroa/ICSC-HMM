function [] = animframetraj(t, scale, foldername, basename)
%ANIMFRAMETRAJ  animates a series of frames
%
%	ANIMFRAMETRAJ(T)
%
% T is a series of homogeneous transforms.  This function animates the
% display of them.
%
% See also: DRAWFRAMETRAJ, DRAWFRAME.

% $Id: animframetraj.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  if ~ishom(t(:,:,1)),
    error('SCREWS:animframetraj', 't is not a homogeneous transform');
  end
  
  if ~exist('scale', 'var'),
    scale = 1;
  end
  
  if 4 == nargin,
    save_movie = true;
  else
    save_movie = false;
  end
    
  hchek = ishold;
  hold on  
  
  n = size(t, 3);  
  
  % initialize
  X = [];
  Y = [];
  Z = [];
  
  % draw the path
  for j=1:n,
    T = t(:,:,j);    
    X = [X;(T * [scale;0;0;1])']; % for the x axis
    Y = [Y;(T * [0;scale;0;1])']; % for the y axis
    Z = [Z;(T * [0;0;scale;1])']; % for the z axis
    
    p = pos(T);
    
    if 1 == j,
      xline = line([p(1);X(j,1)],[p(2);X(j,2)],[p(3);X(j,3)], ...
              'LineStyle', '-', 'color', [1 0 0]);
      yline = line([p(1);Y(j,1)],[p(2);Y(j,2)],[p(3);Y(j,3)], ...
              'LineStyle', '-', 'color', [0 1 0]);
      zline = line([p(1);Z(j,1)],[p(2);Z(j,2)],[p(3);Z(j,3)], ...
              'LineStyle', '-', 'color', [0 0 1]);
      
      set(xline,'Erase','xor', 'LineWidth', 2); 
      set(yline,'Erase','xor', 'LineWidth', 2); 
      set(zline,'Erase','xor', 'LineWidth', 2);                   
    else
      line(X(j-1:j,1),X(j-1:j,2),X(j-1:j,3), ...
        'LineStyle', '--', 'color', [0.5 0 0])
      line(Y(j-1:j,1),Y(j-1:j,2),Y(j-1:j,3), ...
        'LineStyle', '--', 'color', [0 0.5 0])
      line(Z(j-1:j,1),Z(j-1:j,2),Z(j-1:j,3), ...
        'LineStyle', '--', 'color', [0 0 0.5])
      
      % move the axes
      set(xline,'XData',[p(1);X(j,1)],...
                'YData',[p(2);X(j,2)],...
                'ZData',[p(3);X(j,3)]);
      set(yline,'XData',[p(1);Y(j,1)],...
                'YData',[p(2);Y(j,2)],...
                'ZData',[p(3);Y(j,3)]);
      set(zline,'XData',[p(1);Z(j,1)],...
                'YData',[p(2);Z(j,2)],...
                'ZData',[p(3);Z(j,3)]);
    end
    nice3d();
    drawnow;
    refreshdata();
    
    
    if save_movie,
      if ~isdir(foldername),
        mkdir(foldername);
      end
      name = sprintf('_%04i.png', j);
      saveas(gcf, strcat(foldername, '/', basename, name));
    else
      pause(0.005);
    end
  end   
 
  if 0 == hchek
     hold off
  end
  
end