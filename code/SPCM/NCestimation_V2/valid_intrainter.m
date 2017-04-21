function [avintra, avinter, intra, inter, nintra, ninter] = valid_intrainter(Smatrix,U)
%caculate intra similarity/distance and inter similarity

NC=length(U);
nintra=zeros(1,NC);
intra=zeros(1,NC);
avintra=zeros(1,NC);
ninter=zeros(NC,NC);
avinter=zeros(NC,NC);

inter=zeros(NC,NC);
for i=1:NC
  ind=U{i};
  ni=length(ind);
  R=Smatrix(ind,ind);
  if ni==1
    nintra(i)=1;
    intra(i)=0.2;                 %single element similarity:?max(max(Smatrix))/2
    avintra(i)=0.2;
  else
    Q=sum(sum(triu(R,1)));
    T=sum(sum(tril(R,-1)));
    if Q < T
       Q = T;
    end
    intra(i)=Q;                             %similarity sum of intra cluster i
    if ni < 2
       nintra(i) = 1;
    else
       nintra(i)=(ni*(ni-1))/2;        %number of mates in half matrix
    end
    avintra(i)=intra(i)/nintra(i);     %average similarity in cluster i
  end
  
  for j=i+1:NC
    indj=U{j};
    nj=length(indj);
    R=Smatrix(ind,indj);
    inter(i,j)=sum(sum(R));          %disimilarity between cluster i & j
    ninter(i,j)=ni*nj;                       %number of non mates between
    if ninter(i,j) == 0
       ninter(i,j) = 1;
    end
    avinter(i,j)=inter(i,j)/ninter(i,j); %average similarity
  end
  
end
