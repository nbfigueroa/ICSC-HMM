function [R, dmax]= similarity_euclid(data)
% input:    data --- observations x dimensions
% output: R --- nrow * nrow matrix with all the pairwise Euclidean distances
%             between nrow observations in the dataset.

nrow = size(data,1);
R=zeros(nrow,nrow);
data = data';
dmax=0;

% distance between two observations
for i=1:nrow-1
 x=data(:,i);
 for j=i+1:nrow
   y=x-data(:,j);
   d=y'*y;
   d=sqrt(d);
   R(i,j) = d;
   R(j,i) = d; 
   if d>dmax 
       dmax=d; 
   end
 end
end
