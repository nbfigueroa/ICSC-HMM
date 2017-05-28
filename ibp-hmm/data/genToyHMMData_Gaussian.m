function [Data, True_states] = genToyHMMData_Gaussian(N, display)
% INPUTS ----------------------------------------------------------
%    nStates = # of available Markov states
%    nDim = number of observations at each time instant
%    N = number of time series objects
%    T = length of each time series
% OUTPUT ----------------------------------------------------------
%    data  :  SeqData object


Data   = [];
states = [];

for iter = 1:N    
    X1 = mvnrnd([0,2], [0.3, -0.2; -0.2, 0.5]/5, 30);
    X2 = mvnrnd([0,0], [0.5, 0.2; 0.2, 0.3]/5, 20);
    X3 = mvnrnd([0,4], [0.5, 0; 0, 0.3]/5, 40);
    X = [X1; X2; X3];
    labels = [ones(1,30) 2*ones(1,20) 3*ones(1,40)];
    Data{iter}   = X;
    True_states{iter} = labels';
end
label_range = [1 2 3];

if display == 1
    figure('Color',[1 1 1])
    for i=1:length(Data)
        plot(Data{i},'-.', 'LineWidth',2,'Color',[rand rand rand]); hold on
        grid on
    end
    axis( [1 length(X) min(min(X)) max(max(X))] );
    xlabel('Time (1,...,T)')
    ylabel('$\mathbf{x}$','Interpreter','LaTex')
    legend({'O_1','O_2','O_3'})
end

if display == 2
    ts = [1:length(Data)];
    figure('Color',[1 1 1])
    for i=1:length(ts)
        X = Data{ts(i)};
        true_states = True_states{ts(i)};
        
        % Plot time-series with true labels
        subplot(length(ts),1,i);
        data_labeled = [X true_states]';
        plotLabeledData( data_labeled, [], strcat('Time-Series (', num2str(ts(i)),') with true labels'), {'x_1','x_2'}, label_range)
    end
end


end % main function



