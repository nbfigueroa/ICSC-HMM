function [barM sum_w] = sample_barM(M,beta_vec,rho0)

barM = M;
sum_w = zeros(size(M,2),1);

for j=1:size(M,2)
    if rho0>0
        p = rho0/(beta_vec(j)*(1-rho0) + rho0);
    else
        p = 0;
    end
    sum_w(j) = randbinom(p,M(j,j));
    barM(j,j) = M(j,j) - sum_w(j);
end

return;