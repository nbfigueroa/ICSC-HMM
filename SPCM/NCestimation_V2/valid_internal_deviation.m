function [DB,CH,KL,Han,st] = valid_internal_deviation(data,labels,dtype)
% cluster validity indices based on deviation

[nrow,nc] = size(data);
labels = double(labels);
k=max(labels);
if dtype == 1
   [st,sw,sb,cintra,cinter] = valid_sumsqures(data,labels,k);
else
   [st,sw,sb,cintra,cinter] = valid_sumpearson(data,labels,k);
end
ssw = trace(sw);
ssb = trace(sb);

if k > 1
% Davies-Bouldin
  R = zeros(k);
  dbs=zeros(1,k);
  for i = 1:k
    for j = i+1:k
      if cinter(i,j) == 0 
         R(i,j) = 0;
      else
         R(i,j) = (cintra(i) + cintra(j))/cinter(i,j);
      end
    end
    dbs(i) = max(R(i,:));
  end
  DB = mean(dbs(1:k-1));
  
  CH = ssb/(k-1); 
else
  CH =ssb; 
  DB = NaN;
  Dunn = NaN; 
end

CH = (nrow-k)*CH/ssw;    % Calinski-Harabasz
Han = ssw;                        % component of Hartigan
KL = (k^(2/nc))*ssw;         % component of Krzanowski-Lai
