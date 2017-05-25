function [ ] = plotSegmentedData( Xn_seg, seq , Total_feats, Segm_results, my_color_map)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

if isempty(my_color_map)
    % Create color maps
    custom_color_map = [1 1 0; 1 0 1; 0 1 1; 1 0 0; 0 1 0; 0 0 1; 1 1 1; 0 0 0; 1 1 0.5; 0.5 0.5 1; 1 0.5 0.5; rand rand rand; rand rand rand;rand rand rand;rand rand rand;rand rand rand
        rand rand rand;rand rand rand;rand rand rand];

    c = [];
    for i=1:max(Total_feats)
        c(i,:) = custom_color_map(i,:);
    end
else
    
    c = my_color_map;
end
% Plot Segmented Data
for j=1:length(seq)
    
    TrajSeg = [];
    TrajSeg = Segm_results{seq(j),1}
    Traj = Xn_seg{seq(j),1};
    SegPoints = [1; TrajSeg(:,2)];
    for i=1:length(TrajSeg)
        plot3(Traj(1,SegPoints(i):SegPoints(i+1)),Traj(2,SegPoints(i):SegPoints(i+1)),Traj(3,SegPoints(i):SegPoints(i+1)),'Color', c(TrajSeg(i,1),:),'LineWidth', 3);
        hold on
        
        % Plot Starting and End Points
        start_points = [Traj(1,1), Traj(2,1),Traj(3,1)];
        end_points   = [Traj(1,end),Traj(2,end),Traj(3,end)];
        scatter3(start_points(:,1),start_points(:,2),start_points(:,3), 70, [0 1 0], 'filled'); hold on;    
        scatter3(end_points(:,1),end_points(:,2),end_points(:,3),70, [1 0 0], 'filled'); hold on;
    end   
     
    xlabel('x');ylabel('y');zlabel('z');
    axis equal
    grid on
end


end

