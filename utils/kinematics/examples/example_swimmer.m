% This example shows using the kinematic functions for generating the
% magnetic control fields for our artificial bacterial flagella robots.

clear;
close all;

% control variables

theta = 1; % our rotation speed [Hz]
beta = pi/4; % [rad]
gamma = pi/4; % [rad]

b = 0.1; % [T] field strength

elapsed_time = 2; % [s]
delta = 0.05; % time resolution [s]

% set to true if you want to drive the robot
interactive = false;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

omega = [0;0;0;1;0;0]; % our rotation axis for the magnetic field

v = [1; 0; 0; 0; 0; 0;]; % assume we move forward along the x axis [m/s]
                         % this is to simulate real motion

H1 = expm(twist(2*pi*theta*omega*delta)); % this is in a local coordiate
                                          % frame based on the last 
                                          % commanded orientation
H2 = expm(twist([0; 0; 0; 0; beta; 0])); % steering direction
H3 = expm(twist([0; 0; 0; 0; 0; gamma])); % steering direction
H4 = expm(twist(v*delta)); % this is to simulate motion in the real world
           
R = eye(4);

ix = 0;
for i=0:delta:elapsed_time,
  
  ix = ix + 1;

  H2 = expm(twist([0; 0; 0; 0; beta; 0])); % steering direction
  H3 = expm(twist([0; 0; 0; 0; 0; gamma])); % steering direction

  R = R*H1; 
  if (1 == ix),
    H(:,:,ix) = H2*H3*R;
  else
    H(:,:,ix) = H2*H3*H4*R;  
    H(1:3, 4, ix) = H(1:3, 4, ix) + H(1:3, 4, ix-1);
  end  
  

  % this is to allow the user to interactively steer the robot (mostly)
  % with the keyboard as we run.
  if exist('getkeywait', 'file') && interactive,
    fprintf('beta: %0.3f, gamma: %0.3f\n', beta, gamma)
    
    key = getkeywait(delta);

    switch(key)
      case {49}
        beta = beta - 0.1;
      case {50}
        beta = beta + 0.1;
      case {51}
        gamma = gamma - 0.1;
      case {52}
        gamma = gamma + 0.1;
      case {113}
        fprintf('received stop\n');
        break;
      case {-1}
      otherwise
        fprintf('unknown keypress.  please press 1,2,3,4, or Q\n');
    end
   
    
    named_figure('driving');
    hold on;
    drawframe(H(:,:,ix), 0.1);
    nice3d();
  end     

end

%% display the final output 
named_figure('helical path');
clf;
drawframetraj(H, 0.1);
nice3d()
hold on;


% named_figure('animation');
% clf;
% animframetraj(H, 1.0, '/tmp', 'swimmer_movie');

