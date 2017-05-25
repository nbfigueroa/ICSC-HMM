% A simple set of tests for the forward and inverse kinematics using the
% toolbox.

clear;
close all;

fprintf('starting robotlinks test\n');

% debuglevel
% 'none'     0
% 'low'      1
% 'normal'   2
% 'high'     3

global DebugLevel;
DebugLevel = 1;

% % scara example from Murray
% l1 = createtwist([0;0;1], [0;0;0]);
% l2 = createtwist([0;0;1], [0;1;0]);
% l3 = createtwist([0;0;1], [0;2;0]);
% l4 = createtwist([0;0;0], [0;0;1]);
% M  = eye(4);

l1 = [0;0;0;0;0;1];
l2 = [0;10;-10;1;0;0];
l3 = [0;1;0;0;0;0];
M  = eye(4);
M(2,4) = 2;

% % random
% l1 = randtwist();
% l2 = randtwist();
% l3 = randtwist();
% M = randhom();

rn = robot({l1, l2, l3},M);

%% draw a robot

fprintf('starting draw kinematics test...\n');

% trajectory
tn = [-pi/2:pi/16:pi/2];
tz = zeros(size(tn));
tn = [tn;tz;tz];

% draw simple trajectory
hn = fkine(rn,tn);

named_figure('simple trajectory moving all joints');
clf;
drawframetraj(hn);
nice3d();

fprintf('draw kinematics test finished!\n');
%% test the inverse kinematics

fprintf('starting inverse kinematics test...\n');

for i=1:20,
  tic;
  fprintf('  calculating inverse...');
  theta = pi/2*rand(3,1) - pi/4;
  pose = twistcoords(logm(fkine(rn, theta)));
  guess = zeros(3,1) + 0.1*rand(3,1);

%   theta_hat = ikine(rn, pose, guess);
  theta_hat = ikine2(rn, pose, guess);
  fprintf(' completed in %0.3f seconds\n', toc);
  
  if ~isequalf(theta, theta_hat, 1e-3),
    fprintf(' correct solution not found\n');
    display([theta theta_hat]);
    error(' problem with inverse kinematics');
  end  
end

fprintf('inverse kinematics test finished!\n');
%% more inverse kinematics

fprintf('starting bulk inverse kinematics test...\n');

ha = fkine(rn, tn);

fprintf('  calculating %i inverses...', size(tn,2));
tic;
% ii = ikine(rn, ha);
ii = ikine2(rn, ha);
fprintf(' completed in %0.3f seconds\n', toc);

if ~isequalf(ii, tn, 1e-3),
  error(' problem with inverse kinematics');
end   

fprintf('bulk inverse kinematics test finished!\n');

%%
fprintf('test robotlinks success!!\n\n');







