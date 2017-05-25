function [  ] = plotEEActionInstances(action_data, action_sequence)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
N = length(action_data);
for j=1:length(action_sequence)    
    figure('Color', [1 1 1])
    action = j;
    for i=1:N
        action_ts = action_data{i,action};
        t = 1:length(action_ts);
        subplot(4,1,1)
        plot(t , action_ts(1:3,:)','-', 'LineWidth',0.5);hold on;
        xlabel('t');ylabel('m');
        grid on
        legend('x','y','z')
        
        subplot(4,1,2)
        plot(t , action_ts(4:7,:)','-', 'LineWidth',0.5);hold on;
        xlabel('t');ylabel('-');
        grid on
        legend('q_w','q_i', 'q_j', 'q_k')
        
        subplot(4,1,3)
        plot(t , action_ts(8:10,:)','-', 'LineWidth',0.5);hold on;
        xlabel('t');ylabel('N');
        grid on
        legend('f_x','f_y','f_z')
        
        subplot(4,1,4)
        plot(t , action_ts(10:13,:)','-', 'LineWidth',0.5);hold on;
        xlabel('t');ylabel('Nm');
        grid on
        legend('\tau_x','\tau_y','\tau_z')               
    end    
    suptitle(sprintf('Data from %d instances of Action id: %d',N,action_sequence(action)));
end


end

