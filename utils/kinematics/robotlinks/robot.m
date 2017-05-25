function r = robot(XI, GST, THETA)
%ROBOT  creates a robot struct
%
% OUT = ROBOT(P)
% OUT = ROBOT(XI, GST, THETA)
%
% Creates a robot struct from either P (a parameter vector) or XI, GST, and
% THETA.
%
% See also: ROBOTPARAMS.

% $Id: robot.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil

  r.name = 'none';

  if 1 == nargin, % is it an input vector, or seperate variables?
    % for notational clarity;
    p = XI;    
    % e is the floating point precision we'd like to round to
    % this is used to help get rid of some of the floating point error
    % this rounds at 10e-14
    e = -14;
    
	  r.n = round((size(p,2) - 6)/7);
  
    if size(p,2) ~= 6*r.n+6,
      error('ROBOTLINKS:robot', 'wrong number of parameters');
    end

    for i=1:r.n,
      xi = p(1:6)';
   
      r.xi{i} = xi;
      % remove the extra rows
      p(1:6) = [];
    end

    if (0 ~= exist('roundn'))
      r.g_st = real(roundn(expm(twist(p(1:6)')),e));
    else
      r.g_st = real(expm(twist(p(1:6)')));
    end
    
  else
    
    r.n = max(size(XI));

    if ishom(GST),
      r.g_st = GST;
    else
      error('ROBOTLINKS:robot', 'GST is not a homogeneous transform');
    end

    for i=1:r.n,
      if ~istwist(XI{i})
        error('ROBOTLINKS:robot', 'all elements of XI must be twists!');
      end

      % we store all twists in vector form
      if ~isequal([6 1], size(XI{i})),
        XI{i} = twistcoords(XI{i});
      end

    end

    r.xi = XI;

  end
  
  % this sorts the fields so we can use isequal
  r = orderfields(r);

  if ~isrobot(r),
    error('ROBOTLINKS:robot', 'error creating robot');
  end  
 
end