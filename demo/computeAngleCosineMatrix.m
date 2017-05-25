function Y = computeAngleCosineMatrix(X)

for i=1:size(X,1)
    for j=1:size(X,2)
        Y(i,j) = acosd(X(:,i)'*X(:,j)/(norm(X(:,i))*norm(X(:,j))));
    end
end

end