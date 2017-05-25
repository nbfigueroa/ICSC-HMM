function logSoftEv = calcLogSoftEv(theta, data)
% Calculate soft evidence for particular sequence "data"
%   given the emission parameters theta
% OUTPUT
%  logSoftEv : gives log probability of the t-th observation under theta


Xseq = data;
D = size(Xseq,1);
T = size(Xseq,2);
    
cholInvSigma = chol(theta.invSigma );
logDetInvSigma = 2*sum( log( diag( cholInvSigma) ) );
XdiffMu = bsxfun(@minus, Xseq, theta.mu );
U = XdiffMu'*cholInvSigma';
logSoftEv = 0.5*logDetInvSigma - 0.5*sum( U.^2,2);
logSoftEv = logSoftEv - 0.5*D*log(2*pi);

end