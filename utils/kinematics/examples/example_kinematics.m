% Show how we can use symbolic variables for some of the toolbox
syms w_1 w_2 w_3 v_1 v_2 v_3 theta real;

% xi = createtwist([w_1; w_2; w_3], [p_1; p_2; p_3]);
xi = [v_1; v_2; v_3; w_1; w_2; w_3];


fprintf('twist in se(3)\n');
display(twist(xi));

fprintf('twist in SE(3)');
display(simplify(twistexp(xi, theta)));

fprintf('adjoint');
display(collect(ad(twistexp(xi, theta))));
