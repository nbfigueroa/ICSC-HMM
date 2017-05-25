function [angles X] = computeShapeRepresentation(D,V)
d = size(D,1);
n = d-1;
r = 2;
l = nchoosek(n,r);
angles = zeros(d,l);

X = V*D^1/2;
% X = D^1/2*V';
% X = V*D;
%X = D^1/2;

id_vec = 1:d;
id_mat = nchoosek(id_vec,r);

for i=1:d
    id_mat_i = id_mat(any(id_mat==i,2),:);
    ids_i = id_mat_i;
    ids_i(ids_i==i)=[];
    relVs = [];
    for j = 1:length(ids_i)
        Xi = X(:,i);
        Xj = X(:,ids_i(j));
        relVs(j,:) = (Xi-Xj)';     
    end 
    relAnglerows = nchoosek(1:n,r);
    for k=1:size(relAnglerows,1)
            xi = relVs(relAnglerows(k,1),:);
            xj = relVs(relAnglerows(k,2),:);
%             angles(i,k) = acosd((xi*xj')/(norm(xi)*norm(xj)));
            angles(i,k) = acos((xi*xj')/(norm(xi)*norm(xj)));
    end
end