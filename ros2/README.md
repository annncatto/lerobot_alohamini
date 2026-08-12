# AlohaMini ROS 2 kinematics and MoveIt 2

Target platform: Ubuntu 22.04 with ROS 2 Humble.

The DLS node publishes TF and candidate IK trajectories without owning a motor
serial port. MoveIt provides separate plan-only, mock, and guarded single-arm
hardware paths.

```bash
python scripts/sync_alohamini_kinematics_assets.py --check

source /opt/ros/humble/setup.bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
colcon --log-base ros2/.colcon/log build \
  --base-paths ros2 \
  --build-base ros2/.colcon/build \
  --install-base ros2/.colcon/install \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source ros2/.colcon/install/setup.bash

ros2 launch alohamini_description view_kinematic_model.launch.py
ros2 run alohamini_kinematics ik_dry_run

# Default safety path: KDL + OMPL + collision checking, execution disabled.
ros2 launch alohamini_moveit_config plan_only.launch.py

# Explicit mock execution only; publishes simulated /joint_states.
ros2 launch alohamini_moveit_config mock_execution.launch.py

# Real arm, read-only: opens exactly this port, publishes measured selected-arm
# state, rejects every trajectory, and never enables torque or writes a goal.
ros2 launch alohamini_control hardware_execution.launch.py \
  port:=/dev/ttyACM1 side:=right execute_hardware:=false use_rviz:=false

# Real execution requires this explicit second launch mode. Do not enable it
# until the arm is supported, all seven motors start torque-disabled, and the
# read-only state/mapping has been checked.
ros2 launch alohamini_control hardware_execution.launch.py \
  port:=/dev/ttyACM1 side:=right execute_hardware:=true
```

The explicit Python path prevents an active Conda environment from making
Humble's Python 3.10 packages load under an incompatible interpreter. The
node also needs the current LeRobot source installed, or `repo/src` present
in `PYTHONPATH`. Colcon output is intentionally contained in `ros2/.colcon/`
so ROS builds do not create `build/`, `install/`, and `log/` at the repository
root.

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

The MoveIt description keeps the CAD visual meshes, uses reviewed Moving_Jaw
VHACD pieces, retains the exact base collision mesh, and generates conservative
2 mm-padded boxes for the remaining high-poly CAD collision meshes. This avoids
multi-minute FCL startup while preserving a conservative planning baseline.

`plan_only.launch.py` hard-codes `allow_trajectory_execution=False`. The mock
bridge validates exact joint names, position limits, monotonic timestamps,
segment velocity, acceleration, and start-state tolerance before accepting a
trajectory. The hardware bridge keeps Humble/rclpy (Python 3.10) separate from
the LeRobot motor worker (Python 3.12), takes an advisory per-port ownership
lock, and only controls one selected arm. Its worker latches the measured
position before enabling arm torque, limits each command to two encoder ticks
per 40 ms by default, checks current and torque every cycle, and disables only
the six arm motors it enabled at completion, cancellation, or error. The
unconnected robot joints published for MoveIt's full-state requirement are
fixed defaults, not hardware measurements.
