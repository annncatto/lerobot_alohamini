# AlohaMini ROS 2 IK (dry-run foundation)

Target platform: Ubuntu 22.04 with ROS 2 Humble.

The current packages are P0-P2 only. They publish TF and candidate IK trajectories;
they do not own motor serial ports and cannot execute hardware actions.

```bash
python scripts/sync_alohamini_kinematics_assets.py --check

source /opt/ros/humble/setup.bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
colcon build --base-paths ros2 --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch alohamini_description view_kinematic_model.launch.py
ros2 run alohamini_kinematics ik_dry_run
```

The explicit Python path prevents an active Conda environment from making
Humble's Python 3.10 packages load under an incompatible interpreter. The
node also needs the current LeRobot source installed, or `repo/src` present
in `PYTHONPATH`.

The IK node requires a complete measured `/joint_states` message and accepts targets at:

```text
/alohamini/left/target_pose   frame_id=left_Base
/alohamini/right/target_pose  frame_id=right_Base
```

Successful solutions are published only as non-executing candidates:

```text
/alohamini/left/candidate_joint_trajectory
/alohamini/right/candidate_joint_trajectory
```
