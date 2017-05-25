function out = robotparams(r)
%ROBOTPARAMS  returns a parameter vector from a robot
%
% OUT = ROBOTPARAMS(R)
%
% Returns a vector of the robot parameters.
%
% See also: ROBOT.

% $Id: robotparams.m,v 1.1 2009-03-17 16:40:18 bradleyk Exp $
% Copyright (C) 2005, by Brad Kratochvil


  if ~isrobot(r),
    error('ROBOTLINKS:robotparams', 'r is not a robot');
  end

  out = [];
  for i=1:r.n,
    out = [out; r.xi{i}];
  end
  
  
  g_twist = real(logm(r.g_st));
  out = [out; g_twist(1:3, 4); g_twist(3,2);g_twist(1,3);g_twist(2,1)]';
  
  
  % check to make sure everything is a number    
  if ~isempty(out(isnan(out))) || ~isempty(out(isinf(out))),
    error('ROBOTLINKS:robotparams', 'a parameter is NaN or inf');
  end  
  
end