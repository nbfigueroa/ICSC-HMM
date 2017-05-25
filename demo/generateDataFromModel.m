function Y = generateDataFromModel(M)

Y = [];
y_t = [];
y_init = M.data(:,1);
% y_init = ones(8,1);
Y = [Y y_init];
mu = zeros(8,1);


for i=1:length(M.data)
    if i==1
        y_t_1 = y_init;
    end  
    Sigma = M.invSigma/max(max(M.invSigma));
%     Sigma = M.invSigma;
%     e_t =  mvnrnd(mu, Sigma)';
%     y_t = M.A*y_t_1 + e_t;
    y_t = M.A*y_t_1;
    y_t_1 = y_t;
    Y = [Y y_t];
end

end