In the data_collect_july2015_uniroma1 folder there are the following sub-folders:

1) bvh: Here there are the 5 MoCap files captured via the XSens suite, with the 5 corresponding actions performed by the operator in the CK-12 environment. The actions are: RemovingGuard, Cleaning (Brushing), RemovingPanel, Changing Belt, Placing Panel. Each file represents an action sequence, providing the transformation matrices for each skeleton joint along time. It is possible to import and analyze the data in Matlab using a Mocap Toolbox (like the Lawrence Toolbox) or simply to visualize the data in Blender (File -> Import Bvh).

2) pcl: Here you can find the pointclouds of the CK-12 environment acquired by the Kinect either with the ladders (scene_with_ladders.txt) or without ladders (Scene_without_ladders.txt). The two different ladders are also provided separately (ladder1.txt and ladder2.txt respectively). These pointclouds can be visualized in CloudCompare.

3) 3ds: Here you can find the 5 3ds files with the 3d model of the operator acting in the static scene of the pointcloud Scene_without_ladders.txt. In each of these files the operator performs one of the 5 actions according to the bvh files described in 1. In order to open these files, 3DS Max is needed (File -> Open ->select a 3ds file). The three folders included in the 3ds folder contain supporting files needed for correctly importing the 3ds files, and therefore you can ignore them.

4) render: These are the 5 rendered movies (.mp4 format) obtained by the previous 3ds files.

5) por (Gaze Machine): These are the five files (.mp4 format) showing the Points Of Regard, mapped in the scene as blue ellipses, acquired with the Gaze machine. The GM is worn by an assisting human which should be replaced by the robot.