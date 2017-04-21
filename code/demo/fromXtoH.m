function [H] = fromXtoH(X,scale)
Xnr=X;
Xnr(4:7,:) = X(4:7,:)/scale; 
data = Xnr;
R = quaternion([data(4,:);data(5,:);data(6,:);data(7,:)], true);
% Checking for NaNs
for i=1:length(R)
    Ri = R(:,:,i);
    if sum(sum(isnan(Ri)))>0
        Rs = cat(3,R(:,:,i-1),R(:,:,i+1));
        Rnew = [mean(Rs(1,1,:)) mean(Rs(1,2,:)) mean(Rs(1,3,:)) ...
                mean(Rs(2,1,:)) mean(Rs(2,2,:)) mean(Rs(2,3,:)) ...
                mean(Rs(3,1,:)) mean(Rs(3,2,:)) mean(Rs(3,3,:))];
        R(:,:,i)=R(:,:,i-1);
    end
end
t = reshape(data(1:3,:),3,1,size(data,2));
norm = [0 0 0 1];
n = repmat(norm',1,size(data,2));
N = reshape(n,1,4,size(data,2));
H_data = cat(2,R,t);
H = cat(1,H_data,N);
end