function [] =  plotObjectData(object_feats, t, titlename)
   
[D N] = size(object_feats);
if isempty(t)
    t = 1:N;
end
    begin_time = t(1);
    figure('Color',[1 1 1])
    subplot(2,1,1)
    plot(t - begin_time,object_feats(1:3,:)')
    legend('\mu_r','\mu_g','\mu_b')
    grid on

    subplot(2,1,2)
    plot(t - begin_time,object_feats(4:6,:)')
    legend('\sigma_r','\sigma_g','\sigma_b')
    grid on
    suptitle(titlename)
    
end