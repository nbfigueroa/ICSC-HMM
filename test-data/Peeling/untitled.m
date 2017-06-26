for i=1:length(proc_data)
    clear X_a X_p
    X_a =  proc_data{i}.active.X;
    X_p =  proc_data{i}.passive.X;  
    Data_noObj{i} = [X_a(:,1:length(X_p)); X_p];
end

%% Visualize time-series to generate human-labels
id = 1;
X = Data_noObj{id};
figure('Color',[1 1 1])
subplot(3,1,1)
plot(X(1:3,:)', 'LineWidth',2)
legend('x','y','z')
grid on

subplot(3,1,2)
plot(X(8:10,:)', 'LineWidth',2)
legend('f_x','f_y','f_z')
grid on


subplot(3,1,3)
plot(X(17:20,:)', 'LineWidth',2)
legend('q_i','q_j','q_k','q_w')
grid on

%%
figure('Color',[1 1 1])
true_states = True_states_noObj{id};
data_labeled = [X(1:13,:) ; true_states];
plotLabeledData( data_labeled, [], strcat('Time-Series (', num2str(id),') with true labels'), [], [1:5])