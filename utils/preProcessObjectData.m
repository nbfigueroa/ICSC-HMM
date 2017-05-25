function [ final_rgb ] = preProcessObjectData(object_feats, options)
    smooth_fact = options.smooth_fact;
    title_name = options.title_name;
    t_a = options.time_arm;
    resampling_factor = length(t_a)/length(object_feats);

    % Smooth-out Signal
    sm_rgb = [object_feats(1:3,:);object_feats(4:6,:)];
    for kk=1:size(sm_rgb,1)
        sm_rgb(kk,:) = smooth(sm_rgb(kk,:),smooth_fact,'moving');
    end

    % Resample to match Hz of other data
    re_rgb = zeros(6,length(t_a));

    for ll=1:size(re_rgb,1)
        resampling_fact = floor(resampling_factor);
        rgb_feat = sm_rgb(ll,:);
        rgb_feat_re = resample(rgb_feat,resampling_fact,1);
        padd = length(t_a) - length(rgb_feat_re);
        rgb_feat_re = [ones(1,floor(padd/2))*rgb_feat(1) rgb_feat_re ones(1,ceil(padd/2))*rgb_feat_re(end)];
        re_rgb(ll,:) = rgb_feat_re;
    end

    final_rgb = [re_rgb(1:3,:);re_rgb(4:6,:)];
    smooth_fact = 0.005;
    for kk=1:size(sm_rgb,1)
        final_rgb(kk,:) = smooth(re_rgb(kk,:),smooth_fact,'moving');
    end

    begin_time = t_a(1);
    figure('Color',[1 1 1])
    subplot(2,1,1)
    plot(t_a - begin_time,final_rgb(1:3,:)')
    legend('\mu_r','\mu_g','\mu_b')
    grid on

    subplot(2,1,2)
    plot(t_a - begin_time,final_rgb(4:6,:)')
    legend('\sigma_r','\sigma_g','\sigma_b')
    grid on

    suptitle(title_name)
    end