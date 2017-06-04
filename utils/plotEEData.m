function [ ] = plotEEData( X, t , titlename)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[D N] = size(X);

if isempty(t)
    t = 1:N;
end

begin_time = t(1);

figure('Color', [1 1 1]);
subplot(4,1,1);
plot(t - begin_time, X(1:3,:)','--');
legend('x','y','z')
grid on;
xlim([1 t(end)])

f_start = 8;
t_start = 11;
subplot(4,1,2);
if D == 13    
    plot(t - begin_time, X(4:7,:)','-');
    legend('q_w','q_i', 'q_j', 'q_k')
else
    plot(t - begin_time, X(4:6,:)','-');
    legend('roll','pitch','yaw')    
    f_start = f_start - 1;
    t_start = t_start - 1;
end
grid on;
xlim([1 t(end)])

subplot(4,1,3);
plot(t - begin_time, X(f_start:f_start+2,:)','-');
legend('f_x','f_y', 'f_z')
grid on;
xlim([1 t(end)])


subplot(4,1,4);
plot(t - begin_time, X(t_start:end,:)','-');
legend('\tau_x','\tau_y', '\tau_z')
grid on;
xlim([1 t(end)])

suptitle(titlename)


end

