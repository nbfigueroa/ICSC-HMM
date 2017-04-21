% Kaijun WANG, sunice9@yahoo.com, Oct. 2006, March 2007

NC = N1:N;
labels = classlabel;
AR = zeros(1,N);
Rand = zeros(1,N);
Mirkin = zeros(1,N);
Hubert = zeros(1,N);
Sil = zeros(1,N);
DB = zeros(1,N);
CH = zeros(1,N);
KL = zeros(1,N);
Ha = zeros(1,N);
Hom = zeros(1,N);
Sep = zeros(1,N);
wtertra = zeros(1,N);

% (1) External validity indices when true labels are known
for i = NC
  [AR(i), Rand(i), Mirkin(i), Hubert(i)] = ...
      valid_RandIndex(labels(:,i),truelabels);
end
if nk > 1
  valid_errorate(labels(:,nk), truelabels);   % error rate if true labels are given
end

Re = strcmp(Rd, 'euclidean');
% (2) Internal validity indices when true labels are unknown
for i = NC
   R = silhouette(data, labels(:,i), Rd);
   Sil(i) = mean(R);        % average Silhouette
   % Davies-Bouldin, Calinski-Harabasz, Krzanowski-Lai
   [DB(i), CH(i), KL(i), Ha(i), ST] = ...
       valid_internal_deviation(data,labels(:,i), Re);
   S = ind2cluster(labels(:,i));
   [Hom(i), Sep(i), wtertra(i)] ...           % weighted inter/intra ratio
       = valid_internal_intra(Dist, S, Re, dmax);
end
  
%  Homogeneity-Separation
Hom = dmax*Hom;
Sep = dmax*Sep;
ER(NC) = (1-sqrt(NC).*NC/nrow);
Hom = ER.*(Hom-Sep);

kl = KL(NC);
ha = Ha(NC);
nl = length(NC);
S = trace(ST);
kl = [S kl];
ha = [S ha];
R = abs(kl(1:nl)-kl(2:nl+1));
S = [R(2: end) R(end)];
kl = R./S;
kl(nl) = kl(nl-1);
R = ha(1:nl)./ha(2:nl+1);
ha = (R-1).*(nrow-[NC(1)-1 NC(1:nl-1)]-1); 
KL(NC) = kl;
Ha(NC) = ha;

% (3) plotting indices
SR = [Rand; AR; Mirkin; Hubert; Sil; DB; CH; KL; Ha; wtertra; Hom];
kfind = [20 20 20 20 2 1 2 2 5 2 2]; 
FR = {'Rand', 'Adjusted Rand', 'Mirkin', 'Hubert', 'Silhouette (Sil)'...
    'Davies-Bouldin (DB)', 'Calinski-Harabasz (CH)', 'Krzanowski-Lai (KL)', ...
     'Hartigan', 'weighted inter-intra (Wint)', 'Homogeneity-Separation'};

valid_index_plot(SR(:,NC), NC, kfind, FR); 
