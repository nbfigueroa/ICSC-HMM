
function alpha = gibbs_conparam(alpha, numdata, numclass, aa, bb, numiter)
%gibbs_conparam     Auxiliary variable resampling of DP concentration parameter

if nargin < 6
  numiter = 1;
end

numgroup   = length(numdata);
totalclass = sum(numclass);

xx = zeros(1,numgroup);

A=zeros(numgroup,2);
A(:,1)=alpha+1;
A(:,2)=numdata';
A=A';

for ii = 1:numiter
  % beta auxiliary variables
  %for jj = 1:numgroup
  %  xj     = dirichlet_rnd([alpha+1 numdata(jj)], 1);
  %  xx(jj) = xj(1);
  %end  
  xj=randdirichlet(A); 
  xx=xj(1,:);
  
  % binomial auxiliary variables
  zz = (rand(numgroup,1).*(alpha+numdata)) < numdata;

  % gamma resampling of concentration parameter
  gammaa = aa + totalclass - sum(zz);
  gammab = bb - sum(log(xx));
  alpha  = randgamma(gammaa) / gammab;
end

