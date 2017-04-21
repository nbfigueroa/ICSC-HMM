function [Hom,Sep,Wintp] = valid_internal_intra(Smatrix,U,dtype,dmax)
% indices base on intra and inter similarity

k = length(U);
nsample = size(Smatrix,1);
ns = k*(k-1)/2;                                    % mates of clusters
if dtype == 1
   Smatrix = 1000-(1000/dmax)*Smatrix;
else
   Smatrix = 1-Smatrix;
end
[avintra, avinter, intra, inter, nintra, ninter] = valid_intrainter(Smatrix,U);

% Homogeneity & Separation
Hom = sum(intra)/sum(nintra);          % average intra similarity
Sep = Hom;
if ns > 0
  Sep = (sum(sum(inter)))/(sum(sum(ninter)));    % average inter
end
if dtype == 1
   Hom = 1-Hom/1000;
   Sep = 1-Sep/1000;
else
   Hom = Hom+Hom -1;
   Sep = Sep+Sep -1;
end

% weighted inter/intra
wintra = zeros(1,k);
Sinter = zeros(1,k);
sinter = zeros(1,k);
Inter = inter+inter';
for i = 1:k
  ind = U{i};
  ni = length(ind);
  Sinter(i) = sum(Inter(i,:))/(nsample-ni);
  sinter(i) = sum(inter(i,:))/(nsample-ni);
  if ni ==1
     ni = 2;
  end
  wintra(i)=2*intra(i)/(ni-1);
end
if k == 2
   sinter(2) = 0.5*Sinter(2);
end
Sintra = sum(wintra);
Sinter = sum(sinter);
Wint = 1-Sinter/Sintra;
Wintp = (1-2*k/nsample)*Wint;    % penalized
