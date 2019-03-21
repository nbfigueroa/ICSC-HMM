%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Data Structure %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dt of data is 0.1
proc_data{i}   == i-th time-series
True_states{i} == labels for each state
proc_data{i}.X  == X \in \mathds{R}^13 with X = [x,y,z,q_i,q_j,q_k,q_w,f_x,f_y,f_z,\tau_x, \tau_y,\tau_z]^T
proc_data{i}.X_dot == X_dot \in \mathds{R}^6 X = [\dot{x},\dot{y},\dot{z},\omega_x, \omega_x,\omega_x]^T

Following might be redundant or not necessary for you
proc_data{i}.H == H \in \mathds{R}^{4\times 4} H is homogeneous transform, include rotation matrix + translation
proc_data{i}.J == H \in \mathds{R}^{7} J are the joint space position of the robot