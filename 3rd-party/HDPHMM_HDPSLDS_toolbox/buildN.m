function [N Nkdot Ns uniqueS] = buildN(z,s,Kz,Ks,Nblock);

N = zeros(Kz,Kz);
Nkdot = zeros(Kz,1);
Ns = zeros(Kz,Ks);
uniqueS = zeros(Kz,1);

for i=1:size(z,2)-1
    N(z(i),z(i+1)) = N(z(i),z(i+1))+1;
    Nkdot(z(i)) = Nkdot(z(i))+1;
end

for kz=1:Kz
    ind_zk = find(z==kz);
    ind_s_zk = ones(length(ind_zk),1)*[1:Nblock] + ((ind_zk'-1)*Nblock)*ones(1,Nblock);
    s_zk = s(ind_s_zk(:));
    uniqueS(kz)=length(unique(s_zk));
    for ks=1:Ks
        Ns(kz,ks) = length(find(s_zk==ks));
    end
end

return;