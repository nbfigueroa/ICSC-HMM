%% Plot of Probability function from spcm for tau = 10
clc
clear all
close all

% Values from H(Theta_i ~ Theta_j) = s(Sigma_i,Sigma_j)
s = linspace(0,5,1000);

% d-Dimension of data samples
dim = [2:1:12];

% Tolerance tau for scaling function
tau = [0:1:10];
tau_idx = length(tau);

% Scaling function
alpha = zeros(length(tau),length(dim));
for i=1:length(tau)
    alpha(i,:) = 10.^(tau(i)*exp(-dim));
end

% Probability Function from SPCM p(theta_i~theta_j|Sigma_i,Sigma_j) = 1/( 1
% + s(Sigma_i,Sigma_j)*alpha(tau,dim))
p_sim = zeros(length(dim),length(s));
for i=1:length(dim)
    p_sim(i,:) = 1./(1+s*alpha(tau(tau_idx),i));
end

figure('Color', [1 1 1])
subplot(2,1,1)
legendinfo = {};
for i=1:length(dim)
    plot(s,p_sim(i,:),'LineWidth', 2, 'Color',[rand rand rand])    
    legendinfo{i} = strcat('dim = ', num2str(dim(i)));
    hold on
end

legend(legendinfo)
ylabel('Probability')
xlabel('SPCM similarity measure')
title('Prob. Function from SPCM')

subplot(2,1,2)
plot(dim,alpha(end,:), 'LineWidth', 3 , 'Color', [1 0 0]);
ylabel('Scaling Factor')
xlabel('Dimensions')
tit = strcat('Scaling. Function from SPCM w. tau = ', num2str(tau(tau_idx)));
title(tit)

%% Plot of Probability function from spcm full
clc
clear all
close all

% d-Dimension of data samples
dim = [2:1:6];

% Values from H(Theta_i ~ Theta_j) = s(Sigma_i,Sigma_j)
s = linspace(0,5,100);

% Tolerance tau for scaling function
taus = linspace(0,10,100);

% Probability Function from SPCM p(theta_i~theta_j|Sigma_i,Sigma_j) = 1/( 1 + s(Sigma_i,Sigma_j)*alpha(tau,dim)figure('Color', [1 1 1])
figure('Color', [1 1 1])
for k=1:1:length(dim)
    dim_idx = k;
    p_sim = zeros(length(s),length(taus));
    for i=1:length(s)
        for j=1:length(taus)
            alpha_ = 10^(taus(j)*exp(-dim(dim_idx)));
            p_sim(i,j) = 1/(1+s(i)*alpha_);
        end
    end
%     c = ones(size(p_sim))*floor(rand*10);
    surfc(s,taus,p_sim)
    hold on
%     ylabel('SPCM similarity value')
%     xlabel('Tolerance Tau')
%     zlabel('Probability')
%     tit = strcat('Probability function for data of dim= ',num2str(dim(dim_idx)));
%     title(tit)
end

alpha(.6)
ylabel('SPCM similarity value')
xlabel('Tolerance Tau')
zlabel('Probability')