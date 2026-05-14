# Gazebo Evaluation Logs

This directory stores compact scoring evidence from local Gazebo evaluations. The full ROS 2/Gazebo workspaces, bag files, Docker layers, and long policy logs are not committed here.

Included scoring files:

- `aic_results_scoring.yaml`: best copied local result, total about 98.82.
- `cc15c_recovery_gui_20260510_103252_scoring.yaml`: earlier recovery result, total about 38.32.
- `cc15c_recovery_20260513_182000_scoring.yaml`: recovery run that loaded successfully but did not complete insertion.

These scores should be read as development evidence. They show that the final heuristic stack could load, perceive, approach, and achieve partial SFP insertion locally, while full reliable seating remained unfinished at the capstone deadline.

