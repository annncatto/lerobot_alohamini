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

For Joy-Con Cartesian teleoperation, the Raspberry Pi Host remains the only
motor-bus owner. Deploy the matching Host/client protocol first, then start
MoveIt in plan-only mode; do **not** start `hardware_execution.launch.py` at the
same time:

```bash
# One-time PC setup for the team's Joy-Con Python package. This does not install
# its optional DKMS driver or change hidraw permissions.
sudo apt-get install libhidapi-hidraw0 libhidapi-libusb0
git clone https://gitee.com/box2ai-robotics/joycon-robotics.git ~/joycon-robotics
python -m pip install -e ~/joycon-robotics

# Raspberry Pi, from the synchronized repository (restart after updating):
python -m lerobot.robots.alohamini.alohamini_host --robot_model alohamini2pro

# PC terminal 1:
source /opt/ros/humble/setup.bash
source ros2/.colcon/install/setup.bash
ros2 launch alohamini_moveit_config plan_only.launch.py \
  use_rviz:=true joycon_preview:=true

# PC terminal 2, dry-run default (never sends an action):
python scripts/joycon_cartesian_control.py \
  --remote-ip ROBOT_PI_IP --robot-model alohamini2pro --side right

# Only after checking measured joints/TCP and IK direction:
python scripts/joycon_cartesian_control.py \
  --remote-ip ROBOT_PI_IP --robot-model alohamini2pro --side right --execute
```

Before starting, pair/connect the right Joy-Con and confirm that it appears as
a Nintendo HID device (`vendor_id=057e`). If it is present but inaccessible,
install the package's narrow udev rule for that VID/PID; do not grant global
read/write access to every `/dev/hidraw*` device.

The Cartesian controller asks the Host for state-only observations, seeds every
IK request from the latest measured arm state, and obtains the Host's live motor
normalization metadata before converting URDF radians back to LeRobot actions.
It never performs the old script's automatic startup Home move. Stick X/Y moves
the TCP in the selected arm base frame; hold `R` and use stick up/down for TCP Z.
The orientation is latched, and `Home` only re-latches the latest measured TCP.
Press `Plus` to toggle orientation mode: stick up/down changes local pitch,
stick left/right changes local yaw, and `SL`/`SR` changes local roll. Press
`Plus` again to return to position mode.
The optional `--avoid-collisions` remains off until the provisional whole-link
AABBs have been replaced with reviewed collision geometry.

`joycon_preview:=true` disables the fixed Home joint publisher and opens a
minimal RViz without the MoveIt MotionPlanning panel. The IK worker publishes
virtual candidate arm joints plus integrated base, lift and gripper commands to
`/joint_states`, so dry-run input is visible without sending robot actions.
The base buttons follow the team's original semantics: `X/B` forward/backward,
`A/Y` left/right, and `R+A/Y` left/right rotation. LeRobot body `+x/+y` means
forward/left, while the CAD-derived URDF root uses `-Y/+X` for forward/left;
the preview performs this explicit 90-degree coordinate conversion. The base
pose is command integration for interface checking, not wheel odometry.
The `JoyCon TCP` display shows the real measured TCP as a large translucent cyan
sphere (it intentionally stays still in dry-run), the latest attempted target
(including a rejected step) as a yellow sphere/line, and the last accepted IK
candidate as a green sphere/line. A yellow/green split therefore identifies an
IK rejection directly instead of making the joystick appear unresponsive. The
target orientation uses red/green/blue local XYZ arrows. Press `Home` to clear
both trace lines and re-latch the current pose.
Lift targets are clamped to the Host-provided soft range and shown in the
controller status line. These are software limits based on accumulated encoder
motion; they do not replace physical end switches or a valid homing reference.

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
