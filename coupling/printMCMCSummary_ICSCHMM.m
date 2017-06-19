function [] = printMCMCSummary_ICSCHMM( iter, Psi, logPr, algParams )

% fprintf( '\t % 5d/%d after %6.0f sec | logPr % .2e | K %2d | K_z %2d | a_g, b_g %2.2f | a_a, b_a %2.2f var_a %2.2f | a_k, b_k %2.2f var_k %2.2f \n', ...
%      iter, algParams.Niter, toc, logPr.all, size(Psi.F,2), Psi.K_z, Psi.bpM.prior.a_mass, Psi.TransM.prior.a_alpha, ...
%      algParams.HMM.var_alpha, Psi.TransM.prior.a_kappa, algParams.HMM.var_kappa);

fprintf( '\t % 5d/%d after %6.0f sec | logPr % .2e | K %2d | K_z %2d \n', ...
     iter, algParams.Niter, toc, logPr.all, size(Psi.F,2), Psi.K_z); 
 
end