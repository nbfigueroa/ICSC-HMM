% This file runs the toolbox through a series of tests.  It's good to
% verify the functionality of the toolbox when individual functions are
% changed.

clear;
%%
global DebugLevel;
DebugLevel = 0;

fprintf('starting screws test\n');

syms b_1 b_2 b_3 real;

a = [1;0;0];
b = [b_1;b_2;b_3];

a_hat = skew(a);
b_hat = skew(b);

% test pack
c = skewcoords(a_hat);

% random test
e = randskew();

e_hat = skew(e);
f = skewcoords(e_hat);

%% equality tests

if ~isequalf(a, c),
  error('problem with skew pack/unpack');
end

if ~isequalf(e, f);
  error('problem with skew pack/unpack');
end

if ~isequal(a_hat', -a_hat),
  error('problem with skew');
end

if ~isequal(b_hat', -b_hat),
  error('problem with skew');
end

if ~isequal(e_hat', -e_hat),
  error('problem with skew');
end

%% Test twists
syms w_1 w_2 w_3 v_1 v_2 v_3 theta real;

o = createtwist([0; 0; 1], [2; 0; 0]);
o_t = createtwist([0; 0; 0], [1; 0; 0]);
p = createtwist([w_1; w_2; w_3], [v_1; v_2; v_3]);
p_t = createtwist([0; 0; 0], [v_1; v_2; v_3]);
q = randtwist('s');
q_t = randtwist('t');

o_hat = twist(o);
q_hat = twist(q);

if o ~= twistcoords(o_hat);
  error('problem with twist pack/unpack');
end
if ~isequalf(q,twistcoords(q_hat));
  error('problem with twist pack/unpack');
end

%% Test matrix exponentials

% skews
e_a = skewexp(a_hat, 1);
e_e = fast_skewexp(skewcoords(e_hat));

% check against matlab expm function
if ~isequalf(e_a, expm(a_hat)),
  error('problem with skewexp operator');
end

if ~isequalf(e_e, expm(e_hat)),
  error('problem with skewexp operator');
end

%% twists
e_o_r = twistexp(o_hat, pi);
e_o_t = twistexp(twist(o_t), pi);

e_q_r = twistexp(q_hat, 1);
e_q_t = twistexp(twist(q_t), 1);

% check against matlab expm function

if ~isequalf(e_o_r, expm(pi*o_hat)),
  error('problem with twistexp operator');
end

if ~isequalf(e_o_t, expm(twist(pi*o_t))),
  error('problem with twistexp operator');
end

if ~isequalf(e_q_r, expm(q_hat)),
  error('problem with twistexp operator');
end

if ~isequalf(e_q_t, expm(twist(q_t))),
  error('problem with twistexp operator');
end
 
%% Test the adjoint operator

X = e_q_r;
Y = e_q_t;

% we do this numerically because symbolically takes too much time.
if ~isequalf(inv(ad(X)), ad(inv(X))),
  error('problem with inverting adjoint operator');
end

if ~isequalf(inv(ad(Y)), ad(inv(Y))),
  error('problem with inverting adjoint operator');
end

if ~isequalf(ad(X)*ad(X), ad(X*X)),
  error('problem with adjoint operator associativity');
end

if ~isequalf(ad(Y)*ad(Y), ad(Y*Y)),
  error('problem with adjoint operator associativity');
end

if ~isequalf(ad(X)*ad(Y), ad(X*Y)),
  error('problem with adjoint operator associativity');
end

%% test the is functions

if ~isskew(e_hat),
  error('problem with isskew');
end

if ~isrot(e_a),
  error('problem with isrot');
end

if ~istwist(o_hat),
  error('problem with istwist');
end

%% Test homtotwist amd twistlog

for j=1:100,
  q_hat = randtwist();

  angle = 0:0.1*pi:2*pi;
  for i=1:size(angle, 2),

    
    h = twistexp(q_hat, angle(i));
    [xi theta] = homtotwist(h);
    [xi_hat] = twistlog(h);

    if ~isequalf(twistexp(xi, theta), h, 6),
      error('homtotwist test failed')
    end
    
    if ~isequalf(twistexp(xi_hat), h, 6),
      error('twistlog test failed')
    end

  end
  
end

%% let's do these tests a few times

for i=1:100,
  % adjoint test

  a = twist(randtwist());
  b = twist(randtwist());
  c = twist(randtwist());

  test1 = ad(expm(a))*ad(expm(b))*ad(expm(c));
  test2 = ad(expm(a)*expm(b)*expm(c));

  if ~isequalf(test1,test2),
    error('adjoint test not the same');
  end

  % multiplication test

  r = rand(3,1);

  test3 = [twistcoords(a) twistcoords(b) twistcoords(c)]*r;
  test4 = twistcoords(a*r(1) + b*r(2) + c*r(3));

  if ~isequalf(test3, test4),
    error('problem with multiplication test');
  end
end

%% test graphics

named_figure('skew');
for i=1:10, 
  clf;
  drawskewtraj(randskew(),0:pi/20:1/2*pi);
  nice3d; 
end

named_figure('twist');
for i=1:10,
  clf;
  drawtwisttraj(randtwist(),0:pi/20:1/2*pi);
  nice3d; 
end

named_figure('3 DOF robot');
clf;
r = robot({randtwist('r');randtwist('t');randtwist('s')},randhom());
fk = fkine(r, [0:pi/50:pi ; 0:pi/25:2*pi ; 0:pi/25:2*pi]);
animframetraj(fk);

% if we want to save a movie
% clf;
% animframetraj(fk, 1.0, '/tmp', 'example_frame_traj_movie');

fprintf('test screws success!!\n\n');




