function [Data] = genToyHMMData_Gaussian(N)
% INPUTS ----------------------------------------------------------
%    nStates = # of available Markov states
%    nDim = number of observations at each time instant
%    N = number of time series objects
%    T = length of each time series
% OUTPUT ----------------------------------------------------------
%    data  :  SeqData object


for i1 = 1:N    
    X1 = mvnrnd([0,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
    X2 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
    X3 = mvnrnd([0,4], [0.5, 0; 0, 0.3]/5, 40);
    X = [X1; X2; X3];
    Data{i1} = X;
end

end % main function



