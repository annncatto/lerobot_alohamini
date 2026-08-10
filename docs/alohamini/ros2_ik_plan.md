# AlohaMini2Pro ROS 2 逆运动学控制计划

## 1. 目标与边界

本分支规划一条可验证、可逐步开放真机权限的末端空间控制链路：

```text
LeRobot policy / teleop / ROS PoseStamped
                    │
                    ▼
        ROS 2 IK node（右臂优先）
                    │ JointTrajectory
                    ▼
 AlohaMini hardware bridge（唯一串口所有者）
                    │ 单位映射、限位、变化率限制、电流保护
                    ▼
                 真机舵机
```

第一版目标：

- 复用 RoboTwin 的 AlohaMini2Pro URDF、DH、POE 和硬件标定资产；
- 使用 ROS 2 标准消息发布关节状态和接收末端目标；
- 使用数值 IK，不为非 UR 构型重新手写解析逆解；
- 先以右臂打通离线验证，再用同一求解器参数化左右臂；两侧均经过 ROS 只读和
  dry-run 后，才分别开放低速真机控制；
- 保持现有 LeRobot 数据集、policy 和 `Robot` 接口默认行为不变。

第一版不做：

- 双臂协同避碰、抓取规划和移动底盘联合优化；
- 直接用 IK 绕过 AlohaMini 的关节变化率、电流和 watchdog 安全层；
- 让 ROS 节点与现有 AlohaMini Host 同时打开相同串口。

## 2. 已有资产结论

资产源目录：

```text
/home/anncatto/RoboTwin/assets/embodiments/alohamini2pro
```

可直接用于离线开发的内容：

- `urdf/alohamini2pro.urdf`：整机运动树；
- `urdf/alohamini2pro_right_arm.urdf`：右臂独立模型；
- `urdf/alohamini2pro.srdf`：已知内部碰撞排除；
- `config/right_arm_kinematics.yaml`：精确标准 DH、POE、base/tool transform；
- `config/hardware_joint_map_right.yaml`：右臂参考 tick、方向和实测安全端点；
- `config/joint_limits.yaml`：仿真限位及其可信度标记；
- `config/kinematics.yaml`：Fixed Jaw、tool axis 和 TCP 偏移。

当前 DH 与右臂 URDF 已在 1000 个随机姿态上验证，最大位置误差
`2.62e-16 m`，最大姿态误差 `4.22e-8 rad`。这证明 DH 与当前 URDF
数学等价，但不等于整机已完成实物几何标定。

左右臂的六个关节 origin、axis 和局部运动链已经逐项比较，URDF 中完全一致。
上游说明及操作者确认表明：现有硬件参数实际只用一条物理手臂测量，随后复用于
左右两侧；旧 `hardware_joint_map_left.yaml` 中看似独立的采集值已经废弃，不能继续
被当成第二套实测标定。

因此 `hardware_joint_map_right.yaml` 是唯一校准真源，名称中的 `right` 仅保留历史
兼容含义。`hardware_joint_map_left.yaml` 是只读别名，通过 `inherits` 固定继承唯一
校准源；左右侧只改变 LeRobot/URDF/ROS 的逻辑名称前缀，不维护两套 tick、sign、
安全端点或夹爪行程。映射加载器禁止对左侧别名单独执行 capture，避免再次产生漂移。

## 3. 技术选择

### 3.1 IK 求解器

优先复用 LeRobot 的 `RobotKinematics`/Placo 数值 IK，并增加 AlohaMini 双臂参数化包装层：

- 关节顺序固定为 RoboTwin `right_arm_kinematics.yaml` 的六关节顺序，左臂只替换
  joint/frame 前缀；
- 末端 frame 分别使用 `left_Fixed_Jaw`、`right_Fixed_Jaw`，同时定义显式 TCP fixed frame；
- 当前观测关节作为每次求解初值，避免解支跳变；
- 返回结构化结果：成功状态、位置误差、姿态误差、限位余量和求解耗时；
- 求解失败、NaN、越限、残差超限时不发布硬件命令。

DH/POE 的用途是提供独立 FK/Jacobian 校验基准，不作为第一版生产 IK 的唯一实现。
后续需要碰撞感知轨迹规划时再接 MoveIt 2 或 cuRobo，而不是把碰撞规划混入第一版
单点 IK。

### 3.2 ROS 版本与依赖

当前开发机是 Ubuntu 22.04，目标 ROS 版本应为 ROS 2 Humble。当前环境状态：

- 尚未安装 `ros2`/`rclpy`；
- 当前 Conda 环境的 Placo 因缺少 `liburdfdom_sensor.so.4.0` 无法导入；
- `pyproject.toml` 已对 Placo/urdfdom ABI 给出版本约束，但现有环境需要重建或修复。

依赖准备必须先在独立环境完成，不能为了 ROS 修改现有训练环境。建议：

- ROS 节点使用系统 ROS 2 Humble Python；
- IK 核心保持纯 Python、ROS 无关，可在 LeRobot 环境单测；
- 通过清晰的可选依赖或进程边界连接 ROS 与 LeRobot；
- CI 中没有 ROS 时，ROS 集成测试显式 skip，核心数学测试仍必须运行。

### 3.3 单位与关节映射

ROS 统一使用弧度和米。当前 AlohaMini 默认手臂状态是 LeRobot 归一化值
`[-100, 100]`，不能直接当作 URDF 角度。

必须实现并测试一个显式映射层：

```text
LeRobot motor key
  ↔ 原始 encoder tick
  ↔ RoboTwin hardware reference/sign
  ↔ URDF q (rad)
  ↔ ROS JointState / JointTrajectory
```

关键名称映射：

| LeRobot | URDF |
|---|---|
| `arm_right_shoulder_pan` | `right_shoulder_pan` |
| `arm_right_shoulder_lift` | `right_shoulder_lift` |
| `arm_right_elbow_flex` | `right_elbow_flex` |
| `arm_right_wrist_flex` | `right_wrist_flex` |
| `arm_right_wrist_yaw` | `right_wrist_yaw_joint` |
| `arm_right_wrist_roll` | `right_wrist_roll` |

夹爪不进入六轴 IK，作为独立标量命令处理。底盘三个自由度和升降轴第一版固定，
不能让单臂 IK 暗中移动底盘或升降轴来获得可达解。

## 4. 建议代码结构

```text
src/lerobot/robots/alohamini/kinematics/
├── joint_mapping.py          # LeRobot/tick/URDF/ROS 单位与名称转换
├── solver.py                 # ROS 无关的双臂参数化 IK 包装与结果类型
└── safety.py                 # 残差、限位、奇异性和连续性检查

ros2/
├── alohamini_description/    # URDF/SRDF/mesh、robot_state_publisher launch
├── alohamini_kinematics/     # PoseStamped → JointTrajectory
└── alohamini_bridge/         # LeRobot Robot 硬件所有者与 ROS 标准接口

scripts/
└── sync_alohamini_ros_assets.py  # 从 RoboTwin 按 manifest/hash 同步资产

tests/
├── robots/test_alohamini_joint_mapping.py
├── model/test_alohamini_kinematics.py
└── ros2/test_alohamini_ros_contract.py
```

RoboTwin 保持资产真源。LeRobot 分支中的 ROS description 必须由同步脚本生成并记录
源文件 SHA256，不能长期手工维护两份不同 URDF。

## 5. ROS 2 接口草案

### 输入

- `/alohamini/{left,right}/target_pose`：`geometry_msgs/PoseStamped`；
- `frame_id` 第一版分别接受 `left_Base`、`right_Base`，之后通过 TF2 支持 `base_link`；
- `/alohamini/{left,right}/gripper_command`：独立夹爪标量或标准 gripper action；
- `/alohamini/enable_motion`：显式使能，启动默认禁止执行。

### 输出

- `/joint_states`：`sensor_msgs/JointState`，位置单位 rad；
- `/alohamini/{left,right}/joint_trajectory`：`trajectory_msgs/JointTrajectory`；
- `/tf`、`/tf_static`：由 `robot_state_publisher` 发布；
- `/diagnostics`：映射状态、IK 残差、限位余量、求解耗时和拒绝原因。

硬件 bridge 是唯一串口所有者。它接收通过安全检查的关节轨迹，将 URDF rad 转回
LeRobot/hardware 单位，组成完整 AlohaMini action，并调用现有 `send_action()`，从而继续
经过关节变化率限制、电流保护和 watchdog。

## 6. 分阶段实施

### P0：资产冻结和依赖可用性

- 建立 RoboTwin → ROS description 的 manifest/hash 同步；
- 修复独立 IK 环境中的 Placo ABI；
- 为 ROS 2 Humble 建立最小 colcon workspace；
- 固定共享 joint order，并明确左右 base frame、Fixed Jaw 和 TCP 定义。

通过条件：资产 hash 可复现，URDF 可由 Placo 和 `robot_state_publisher` 加载。

### P1：纯离线 FK、映射和 IK

- 实现共享单臂校准到左右 LeRobot/URDF 名称的映射；
- 用 DH、POE、URDF/Placo 三路比较左右 FK；
- 从随机安全关节姿态生成可达末端目标，再用 IK 回解；
- 检查多解连续性、限位、奇异区域和不可达目标拒绝。

初始验收建议：

- 映射往返误差不超过一个 encoder tick；
- FK 三路位置误差 `< 1e-6 m`；
- 可达目标 IK 位置残差 `< 1 mm`、姿态残差 `< 1 deg`；
- 所有结果在实测安全限位内；
- 不可达、NaN 和残差超限目标 100% 被拒绝。

### P2：ROS 只读与 dry-run

- bridge 发布双臂 `/joint_states`，但不接受运动命令；
- RViz 中对照已知参考姿态检查模型方向；
- IK node 接收 PoseStamped，只发布候选 JointTrajectory 和 diagnostics；
- 记录目标、当前 q、解 q、FK 回代误差，不写舵机。

通过条件：已知参考姿态、每个关节正方向和 TCP 位姿均由操作者确认。

### P3：低速单臂单点控制

- 默认 motion disabled，需显式使能；
- 每次只接受工作空间内的小位移目标；
- 使用当前实测 q 作为 IK 初值；
- 对 Cartesian step、joint step、joint velocity 和命令年龄分别限幅；
- 超时保持最后安全关节目标，底盘速度强制为零；
- 左右臂分别验收，同一轮测试只使能一侧；
- 先空载、再软环境、最后才进入抓取测试。

通过条件：连续多次小步运动无解支跳变、无电流保护触发、无越限命令。

### P4：LeRobot EE 控制接入

- 增加 ROS client 或 processor，将 LeRobot EE action 发送为 PoseStamped；
- FK 将关节 observation 转成 EE observation；
- 录制 EE-space 数据集时明确 frame、单位和 TCP；
- ACT/Diffusion/VLA 仍可保持原 joint-space 路径，IK 是可选后处理链路；
- chunk action 必须逐步经过 IK 连续性和安全限制，不能只检查 chunk 首尾。

### P5：双臂协同与碰撞规划

- 在两侧各自通过 P1～P3 后才允许同时使能；
- 加入 SRDF、自碰撞和环境碰撞检查；
- 最后再评估 MoveIt 2、cuRobo 以及底盘/升降联合规划。

## 7. 必须始终成立的安全约束

- 同一时刻只有一个进程拥有左右臂串口；
- ROS 消息的 rad 绝不直接写入当前 `[-100, 100]` action 接口；
- IK 成功标志之外还必须检查 FK 回代残差；
- 不使用 URDF 中未标定的占位 limit 作为真机安全范围；
- 共享校准别名无法解析、唯一源 hash 不匹配或左右命名映射不完整时保持 motion disabled；
- 硬件变化率和电流保护位于 IK 下游，对所有模型和 ROS 命令统一生效；
- 通信超时、旧时间戳、frame 不匹配和求解失败都只能导致拒绝/保持，不能发送猜测动作。

## 8. 第一轮开发任务

1. 写资产同步 manifest，复制双臂 URDF 所需 mesh/config 并校验唯一校准源 hash；
2. 修复独立 Placo 环境，运行 RoboTwin 已有 FK 等价测试，并增加左臂对称链测试；
3. 实现双臂 `AlohaMiniJointMapping` 及 tick/rad 往返测试；
4. 实现参数化双臂 `AlohaMiniIKSolver` 和结构化失败结果；
5. 建立 ROS 2 Humble description、joint-state publisher 与 dry-run IK node；
6. 完成操作者只读验收后，另开提交实现低速硬件 bridge。

每一步单独提交；P0～P2 不包含真机写操作，P3 的硬件写入必须作为独立、容易审查和
回退的提交。
