function [ ] =  plotBimanualTrajectories( proc_data )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

figure('Color', [1 1 1])


X_p = proc_data.passive.X; H_p = proc_data.passive.H;
X_a = proc_data.active.X;  H_a = proc_data.active.H;

Passive_robot   = proc_data.passive.base_frame;
Active_robot    = proc_data.active.base_frame;
Cutting_board_l = proc_data.board_lframe;
Cutting_board_r = proc_data.board_rframe;


% Plot Trajectories for Passive and Active Arms
plot3(X_p(1,:),X_p(2,:),X_p(3,:),'Color',[0 0 1], 'LineWidth', 2); hold on;
plot3(X_a(1,:),X_a(2,:),X_a(3,:),'Color',[0 1 0], 'LineWidth', 2); hold on;

% Plot Starting and End Points
start_points = [X_p(1,1), X_p(2,1),X_p(3,1);X_a(1,1),X_a(2,1),X_a(3,1)];
end_points   = [X_p(1,end),X_p(2,end),X_p(3,end);X_a(1,end),X_a(2,end),X_a(3,end)];
scatter3(start_points(:,1),start_points(:,2),start_points(:,3), 100, [0 1 0], 'filled'); hold on;    
scatter3(end_points(:,1),end_points(:,2),end_points(:,3),100, [1 0 0], 'filled'); hold on;

% Draw Important Reference Frames
text(Passive_robot(1,end) + 0.03, Passive_robot(2,end) + 0.03, Passive_robot(3,end) + 0.03,'Passive Robot','FontSize',8, 'Color', [0 0 1]);
drawframe(Passive_robot,0.05)

text(Active_robot(1,end) + 0.03, Active_robot(2,end) + 0.03, Active_robot(3,end) + 0.03,'Active Robot','FontSize',8, 'Color', [0 1 0]);
drawframe(Active_robot,0.05)

text(Cutting_board_l(1,end) + 0.03, Cutting_board_l(2,end) + 0.03, Cutting_board_l(3,end) + 0.03,'Cutting Board-l','FontSize',8, 'Color', [1 0 0]);
drawframe(Cutting_board_l,0.05)

text(Cutting_board_r(1,end) + 0.03, Cutting_board_r(2,end) + 0.03, Cutting_board_r(3,end) + 0.03,'Cutting Board','FontSize',8, 'Color', [1 0 0]);
drawframe(Cutting_board_r,0.05)

% Draw Cutting Board
fill3([Cutting_board_l(1,4), Cutting_board_r(1,4), Cutting_board_r(1,4) - 0.1 , Cutting_board_l(1,4) - 0.1],[Cutting_board_l(2,4), Cutting_board_r(2,4), Cutting_board_r(2,4), Cutting_board_l(2,4)],[0 0 0 0],[1 0 0])

% Draw some frame of Start-end Trajectories
drawframe(H_p(:,:,1),0.05); 
drawframe(H_a(:,:,1),0.05);
drawframe(H_p(:,:,end),0.05); 
drawframe(H_a(:,:,end),0.05);

% Display Legends
legend('Passive Arm', 'Active Arm')
xlabel('x');ylabel('y');zlabel('z') 
title(proc_data.name)
grid on
axis tight


end

