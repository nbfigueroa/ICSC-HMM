z = viterbiZ;
s = maxSgivenZ(likelihood);

N = buildCountMatrix(z);
% Sample M, where M(i,j) = # of tables in restaurant i served dish j:
M = randnumtable([alpha0*beta_vec(ones(1,Kz),:)+kappa0*eye(Kz); alpha0*beta_vec],N);
% Sample barM (the table counts for the underlying restaurant), where
% barM(i,j) = # tables in restaurant i that considered dish j:
[barM sum_w] = sample_barM(M,beta_vec,rho0);
beta_vec = randdirichlet([sum(barM,1) + gamma0/Kz]')';