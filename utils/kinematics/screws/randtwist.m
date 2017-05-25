function xi = randtwist(ch)
%RANDTWIST  generates a pseudo-random twist vector
%
%	XI = RANDTWIST()
% XI = RANDTWIST(CH)
%
% There is an equal chance of pure rotations, pure translations, or screws.
% The values of q are equally distributed from -100 to 100;
%
% CH can be used to select 'r', 't', 's' which stand for rotation, 
% translation, or screw
%
% See also: RANDSKEW.

% $Id: randtwist.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  scale = 5;

  if 1 == nargin,
    switch ch
      case 't',
        s = 0.25;
      case 'r',
        s = 0.50;        
      case 's',
        s = 1;        
      otherwise,
        s = rand;
    end
  else
      s = rand;
  end

  xi = zeros(6,1);
  

  if s < 0.33,  % pure translation
    xi(1:3) = 2*scale*rand(3,1) - scale;    
  elseif s < 0.66, % pure rotation
    xi(4:6) = randskew();    
  else
    q = 2*scale*rand(3,1) - scale;
    xi(4:6) = randskew();    
    xi(1:3) = -skew(xi(4:6))*q;
  end
  
  if ~istwist(twist(xi)),
    error('SCREWS:randtwist', 'error creating twist');
  end
  
end