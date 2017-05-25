function [] = plotPair3dGauss(Theta_i, Theta_j)
    mu_i = Theta_i.mu; Sigma_i = Theta_i.invSigma^-1;
    mu_j = Theta_j.mu; Sigma_j = Theta_j.invSigma^-1;

    [V_i D_i] = eig(Sigma_i);
    figure('Color', [1 1 1])
    [x,y,z] = created3DgaussianEllipsoid(mu_i,V_i,D_i^1/2);
    mesh(x,y,z,'EdgeColor','black','Edgealpha',0.8);
    hold on
    hidden off

    [V_j D_j] = eig(Sigma_j);        
    [x,y,z] = created3DgaussianEllipsoid(mu_j,V_j,D_j^1/2);
    mesh(x,y,z,'EdgeColor','red','Edgealpha',0.8);
    hidden off
    pause(0.1);
    axis equal
end