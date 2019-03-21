function [chosen_K] = hmm_eval(Data,  K_range, repeats)
%HMM_EVAL Implementation of the HMM Model Fitting with AIC/BIC metrics.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o repeats  : (1 X 1), # times to repeat k-means
%       o K_range  : (1 X K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


AIC_mean = zeros(1, length(K_range));
BIC_mean = zeros(1, length(K_range));
AIC_std = zeros(1, length(K_range));
BIC_std = zeros(1, length(K_range));

[numObs dim] = size(Data{1});

% Populate Curves
for i=1:length(K_range)
    
    % Select K from K_range
    K = K_range(i);
    
    % Repeat GMM X times
    AIC_ = zeros(1, repeats); BIC_= zeros(1, repeats);
    numParam = dim*K  + K*K + K + (dim*dim)*K; %param, transition, initial, cov
    
    for ii = 1:repeats
        fprintf('Iteration %d of K=%d\n',ii,i);
        [~, ~, ~, loglik] = ChmmGauss(Data, K);
        
        % Compute metrics from implemented function
        bic_N = - 2 * loglik + numParam * log(numObs);
        aic_N = - 2 * loglik + 2 * numParam;
        
        AIC_(ii) =  aic_N;
        BIC_(ii) =  bic_N;
    end
    
    % Get the max of those X repeats
    AIC_mean(i) = mean (AIC_); AIC_std(i) = std (AIC_);
    BIC_mean(i) = mean (BIC_); BIC_std(i)= std (BIC_);
    
end

% Find optimal value on RSS curve
[~, chosen_K] = ml_curve_opt(BIC_mean,'line');


% Plot Metric Curves
figure('Color',[1 1 1]);
errorbar(K_range',AIC_mean(K_range)', AIC_std(K_range)','--or','LineWidth',2); hold on;
errorbar(K_range',BIC_mean(K_range)', BIC_std(K_range)','--ob','LineWidth',2);
grid on
xlabel('Number of states $K$','Interpreter','LaTex'); ylabel('AIC/BIC Score','Interpreter','LaTex','FontSize',10)
title('Model Selection for Hidden Markov Model','Interpreter','LaTex','FontSize',20)

legend('AIC', 'BIC')



end