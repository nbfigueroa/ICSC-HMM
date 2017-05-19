function numtable=randnumtable(alpha,numdata)

numtable=zeros(size(numdata));
for ii=1:prod(size(numdata))
    numtable(ii)=1+sum(rand(1,numdata(ii)-1)<ones(1,numdata(ii)-1)*alpha(ii)./(alpha(ii)+(1:(numdata(ii)-1)))); 
end;
numtable(numdata==0)=0;