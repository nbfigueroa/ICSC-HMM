function [indx,ssw,sw,sb]=valid_clusterIndex(data,labels)
% clustering validation indices 
% Kaijun WANG, sunice9@yahoo.com, May 2005, Oct. 2006

[nr,nc]=size(data);
k=max(labels);
[st,sw,sb,S,Sinter] = valid_sumsqures(data,labels,k);
ssw=trace(sw);
ssb=trace(sb);

Sil = silhouette(data,labels);
Sil = mean(Sil);             % mean Silhouette

if k>1
  CH = ssb/(k-1);           % Calinski-Harabasz
  %Fish=ssb/ssw; % Fisher;  Han=log10(1/Fish); % Hantigan
  
 % Davies-Bouldin 
  R = NaN * zeros(k);
  dbs=zeros(1,k);
  for i = 1:k
    for j = i+1:k
      R(i,j) = (S(i) + S(j))/Sinter(i,j);
    end
    dbs(i) = max(R(i,:));
  end
  db=dbs(isfinite(dbs));    % Davies-Bouldin for all clusters
  DB = mean(db);             % mean Davies-Bouldin
 
else
  CH =ssb; 
  DB=NaN;
  %Fish=NaN;  Han=0;
end

CH = (nr-k)*CH/ssw;        % Calinski-Harabasz
KL=(k^(2/nc))*ssw;          % Krzanowski and Lai

indx=[Sil DB CH KL]'; 
