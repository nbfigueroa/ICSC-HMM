clc; clear all; close all;
[data, TruePsi, Data, True_states, True_theta] = genToySeqData_TR_Gaussian(4, 2, 3, 500, 0.5 );
title_name  = 'Merging Transformed Emission Parameters';
plot_labels = {'$x_1$','$x_2$'};


%%
figure('Color',[1 1 1]);
True_theta_1.K     = 1;
True_theta_1.Mu    = [1;1];
True_theta_1.Sigma = 0.5*[1.9 -1.3;-1.3 1.9];
plotGaussianEmissions2D(True_theta_1, plot_labels, title_name); hold on;
plotGaussianEmissions2D(True_theta, plot_labels, title_name, [1 2 2 1]);
alpha(0.4)
