% ================================
% function ChmmGaussTest()
% Generate Data for Gaussian HMM test
for i1 = 1:3
    X1 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
    X2 = mvnrnd([0,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
    X3 = mvnrnd([0,4], [0.5, 0; 0, 0.3]/5, 40);
    X = [X1; X2; X3];
    Data{i1} = X;
end
close all


%%
Data{1} = D';
X = D';
figure('Color',[1 1 1])
for i=1:length(Data)
plot(Data{i},'-.', 'LineWidth',2,'Color',[rand rand rand]); hold on
grid on
end
xlabel('Time (1,...,T)')
ylabel('$\mathbf{y}$','Interpreter','LaTex')
% legend({'O_1','O_2','O_3'})


%% Model Selection
dim = size(X,2);
numObs = size(X,1);
bic = [];
aic = [];
logProb_ms = [];
for Q=1:10
    [p_start, A, phi, loglik] = ChmmGauss(Data, Q);
    
    numParam = dim*Q  + Q*Q + Q + (dim*dim)*Q; %param, transition, initial, cov
    bic_N = - 2 * loglik + numParam * log(numObs);
    aic_N = - 2 * loglik + 2 * numParam;
    
    bic = [bic bic_N];
    aic = [aic aic_N];

    logProb_ms = [logProb_ms loglik];
end

%% Visualize plots
figure('Color',[1 1 1])
plot(bic,'-.','LineWidth',2,'Color',[rand rand rand]); hold on;
plot(aic,'-.','LineWidth',2,'Color',[rand rand rand]); hold on;
grid on;
legend({'BIC','AIC'})
title('HMM Model Selection','Interpreter','LaTex','Fontsize',20)
xlabel('Number of states $K$','Interpreter','LaTex')

%% Set feature states
Q = 7;  % state num
p = 12;  % feature dim
p_start0 = [1 0 0];
A0 = [0.8 0.2 0; 0 0.8 0.2; 0 0 1];

[p_start, A, phi, loglik] = ChmmGauss(Data, Q);
% [p_start, A, phi, loglik] = ChmmGauss(Data, Q, 'p_start0', p_start0, 'A0', A0, 'phi0', phi0, 'cov_type', 'diag', 'cov_thresh', 1e-1);

Q
loglik

% Calculate p(X) & vertibi decode
logp_xn_given_zn = Gauss_logp_xn_given_zn(Data{1}, phi);
[~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
path = LogViterbiDecode(logp_xn_given_zn, p_start, A);



% figure('Color',[1 1 1])
% Xall = cell2mat(Data');
% scatter(Xall(:,1), Xall(:,2), '.'); hold on
% plot gaussians
% for k=1:Q
% error_ellipse(reshape(phi.Sigma(:,:,1),p,p), phi.mu(:,1)', 'style', 'r'); hold on
% error_ellipse(reshape(phi.Sigma(:,:,2),p,p), phi.mu(:,2)', 'style', 'g'); hold on
% error_ellipse(reshape(phi.Sigma(:,:,3),p,p), phi.mu(:,3)', 'style', 'k'); hold on
% end
%
data_labeled = [X path]';
plotLabeledEEData(data_labeled, [], strcat('Segmented Data, K:',num2str(Q),', loglik:',num2str(loglik)), 0,{'x_1','x_2'});
