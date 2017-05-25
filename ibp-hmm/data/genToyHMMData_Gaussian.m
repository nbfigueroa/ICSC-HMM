function [data, PsiTrue] = genToyHMMData_Gaussian(N)
% INPUTS ----------------------------------------------------------
%    nStates = # of available Markov states
%    nDim = number of observations at each time instant
%    N = number of time series objects
%    T = length of each time series
% OUTPUT ----------------------------------------------------------
%    data  :  SeqData object


for i1 = 1:N
    X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
    X2 = mvnrnd([0,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
    X3 = mvnrnd([0,4], [0.5, 0; 0, 0.3]/5, 40);
    X = [X1; X2; X3];
    Data{i1} = X;
end

PsiTrue.F = zeros(N, nStates);
for ii = 1:N
    PsiTrue.F(ii, unique( data.zTrue(ii) ) ) = 1;
end
for kk = 1:nStates
    PsiTrue.theta(kk).mu = Px.Mu(kk,:);
    PsiTrue.theta(kk).invSigma = inv( Px.Sigma(:,:,kk) );
end
PsiTrue.Pz = Pz;
PsiTrue.z = zTrue;

end % main function



