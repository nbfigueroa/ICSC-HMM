function [  ] = visualizeRollingEnvironment(Robot_Base, Rolling_Board);
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Draw Important Reference Frames
text(Robot_Base(1,end) + 0.03, Robot_Base(2,end) + 0.03, Robot_Base(3,end) + 0.03,'Robot Base','FontSize',8, 'Color', [0 0 1]);
drawframe(Robot_Base,0.05)

text(Rolling_Board(1,end) + 0.03, Rolling_Board(2,end) + 0.03, Rolling_Board(3,end) + 0.03,'Rolling Board','FontSize', 8, 'Color', [1 0 0]);
drawframe(Rolling_Board,0.05)

% Draw Rolling Board
table_off = 0.02;
table_width = 0.56;
table_height = 0.75;

Table_Origin = Rolling_Board;
Table_Edge1 = eye(4); Table_Edge2 = eye(4); Table_Edge3 = eye(4); Table_Edge4 = eye(4);
Table_Edge1(1:3,4) = [-table_off 0 table_off]'; Table_Edge1 = Table_Origin*Table_Edge1;
Table_Edge2(1:3,4) = [0 0 -table_width]'; Table_Edge2 = Table_Edge1*Table_Edge2;
Table_Edge3(1:3,4) = [table_height 0 -table_width]'; Table_Edge3 = Table_Edge1*Table_Edge3;
Table_Edge4(1:3,4) = [table_height 0 0]'; Table_Edge4 = Table_Edge1*Table_Edge4;
fill3([Table_Edge1(1,4) Table_Edge2(1,4) Table_Edge3(1,4) Table_Edge4(1,4)],[Table_Edge1(2,4) Table_Edge2(2,4) Table_Edge3(2,4) Table_Edge4(2,4)],[Table_Edge1(3,4) Table_Edge2(3,4) Table_Edge3(3,4) Table_Edge4(3,4)],[0.5 0.5 0.5])
    


end

