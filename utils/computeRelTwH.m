function [tw Hrel] = computeRelTwH(H)
    tw = zeros(6,size(H,3));
    Hrel = zeros(4,4,size(H,3));
    for i=1:length(H)
        if i==1
            T = H(:,:,i);
            Hrel(:,:,i) = T;
            [xi theta] = homtotwist(T);
            tw(:,i) = [xi*theta];
        else
            T1 = H(:,:,i-1);
            T2 = H(:,:,i);
            T12 = inv(T1)*T2;
%             T12 = T1\T2;
            Hrel(:,:,i) = T12;
            [xi theta] = homtotwist(T12);
            tw(:,i) = [xi*theta];
        end
    end
    tw(:,1) = zeros(6,1);

end