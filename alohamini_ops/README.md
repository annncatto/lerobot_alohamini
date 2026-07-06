# AlohaMini Ops

这些脚本放在仓库的 `alohamini_ops/`，用于现场启动和调试。

## 推荐启动顺序

1. 确认硬件供电、急停状态、两条 Leader 臂和树莓派 USB 都已连接。
2. 启动树莓派 host：
   ```bash
   alohamini_ops/start_pi_host.sh
   ```
3. 启动本机遥操：
   ```bash
   alohamini_ops/start_teleop.sh
   ```
   这个入口使用 `teleoperate_bi_terminal_keys.py`，直接读取当前终端按键，不依赖 Wayland 下不稳定的 `pynput` 全局键盘监听。
4. 将焦点放在遥操终端里按键：
   - `w/s`: 前进/后退
   - `z/x`: 平移左/右
   - `a/d`: 原地左/右转
   - `u/j`: 升降上/下
   - `r/f`: 速度增/减

## GUI

客户或新电脑首次使用推荐先运行初始化脚本，它会把本机路径、conda 路径、树莓派用户和 IP 写入 `config.env`：

```bash
alohamini_ops/init_customer_env.sh --pi-user pi5 --pi-host 192.168.0.24
alohamini_ops/setup_env.sh
alohamini_ops/start_gui.sh
```

如果不传 `--pi-host`，脚本会从 `ip neigh` / `arp` 列出局域网候选地址，再要求手动输入。初始化脚本只更新机器相关字段，不覆盖相机、模型、任务参数等其它配置。

```bash
python alohamini_ops/main.py
```

GUI 使用 Qt，需要安装 `PyQt6` 或 `PySide6`：

```bash
alohamini_ops/setup_env.sh
```

GUI 结构：

- `main.py`: 入口。
- `app/context.py`: 装配配置、脚本控制器和运行环境。
- `ui/main_window.py`: 主窗口、信号连接、动作调度。
- `ui/ui_manager.py`: 只负责左右分栏和布局。
- `ui/tabs/*.py`: 连接、遥操、校准、诊断功能页。
- `ui/actions/*.py`: 窗口内快捷键和按钮动作。
- `ui/presenters/*.py`: 状态显示刷新。
- `workers/*.py`: 后台命令和 GUI 遥操线程。

GUI 左侧是 `QTabWidget` 控制面板，右侧是状态/视频/地图/传感器预留区域，左下角是日志面板。日志面板支持等级过滤、暂停滚动、清空和保存。

右侧工作区包含：

- `总览`: 机器人、Host、链路、数据采集状态。
- `相机`: 机器人相机接入入口，通过树莓派 Host 的 observation 流接收 `config_lekiwi.py` 中启用的相机，可保存当前帧。
- `地图`: 预留底盘里程计、SLAM、标定坐标系显示。
- `传感器`: 预留电压、电流、关节、底盘、升降状态曲线。

GUI 打开后不会自动连接相机，避免无任务时占用 observation 客户端。启动 GUI 遥操作后，会用同一个遥操作客户端在右侧相机面板显示画面。启动数据采集时，如果独立相机预览正在运行，GUI 会先关闭该预览，再由采集进程打开 Rerun Viewer 显示相机画面。任务启动后的相机显示可在 `config.env` 中通过 `ALOHAMINI_TASK_CAMERA=false` 关闭。

机器人相机配置位于两端同名文件：

```bash
src/lerobot/robots/alohamini/config_lekiwi.py
```

`lekiwi_cameras_config()` 必须在 PC 和树莓派两端保持一致。当前默认启用：

- `forward` -> `/dev/am_camera_forward`
- `wrist_right` -> `/dev/am_camera_wrist_right`

修改后需要同步到树莓派并重启 Host。

Diagnostics 页包含舵机检查：

- `Check Local Leader Servos`: 检查本机 `/dev/am_arm_leader_left` 和 `/dev/am_arm_leader_right`。
- `Check Pi Follower/Base Servos`: 通过 SSH 检查树莓派 `/dev/am_arm_follower_left`、`/dev/am_arm_follower_right`，其中左 bus 也包含底盘 8/9/10 和升降 11。

对应命令行脚本也可单独运行：

```bash
alohamini_ops/check_local_servos.sh
alohamini_ops/check_pi_servos.sh
```

GUI 遥操支持：

- 按钮按住控制底盘/升降，松开停止。
- 窗口内快捷键：`W/S/Z/X/A/D/U/J`，`Space` 停止，`Esc` 急停。
- `Enable global keyboard mapping` 是预留开关，默认关闭；当前优先使用窗口内快捷键，避免 Wayland 全局键盘监听不稳定。
- 长时间命令和遥操循环都在 `QThread`/worker 中执行，不阻塞主界面。

## 数据采集

GUI 的 `数据` 页会在 GUI 后台运行：

```bash
alohamini_ops/start_record.sh
```

该入口继续调用 `lerobot_alohamini/examples/alohamini/record_bi.py`，采集逻辑、数据格式、episode 保存和可选 Hub 上传都与命令行采集保持一致。运行输出会显示在 GUI 左下角日志面板里。

GUI 初始采集流程参考 `ALOHA操作流程分享.pdf` 后半部分，默认采用“先单条采集、检查，再逐条追加”的节奏：

1. 先在 `连接` 页启动树莓派 Host，并确认 Host 正常。
2. 确认 Leader 校准文件存在，且 `config.env` 中的 `LEADER_ID` 与校准时使用的 ID 一致。
3. 不要启动 `GUI 遥操` 或单独 `相机` 预览，采集进程会自己连接机器人、Leader 和键盘。
4. 将机器人、物体和环境摆到第一条演示的初始状态，保持整机重心稳定。
5. 打开 `数据` 页，首次采集建议填写：
   - `数据集 repo_id`: 新名字，例如 `local/alohamini_pick_lift_move_test_01`
   - `采集段数`: `1`
   - `FPS`: `25`
   - `每段时长秒`: `45`
   - `重置时长秒`: `8`
   - `任务描述`: 保持同一任务的数据描述一致
   - 不勾选 `继续写入已有数据集`
6. 点击 `开始数据采集`。
7. 第一条采完后，先用回放或可视化检查数据质量，再继续追加。

采集中仍使用 `record_bi.py` 原有全局按键：

- Leader 主臂：控制左右机械臂
- `W/S/Z/X/A/D`: 底盘移动
- `U/J`: 升降
- 方向键右：提前结束当前段
- 方向键左：丢弃并重录当前段
- `Esc`: 停止整次采集
- GUI 的 `完成当前段并保存`：当前段先写入磁盘；保存完成后进入不记录数据的遥操作复位阶段
- GUI 的 `废弃当前段并等待复位`：当前段会清空，不使用固定 `重置时长秒` 自动倒计时
- GUI 出现复位提示后，仍可继续用 Leader 和键盘遥操作；把机器人和物体带回原处，再点击 `复位完成，继续采集`
- 如需强制停止，点击 GUI 的 `停止数据采集`。强制停止不会保证当前半成品可训练，通常需要换新 repo_id 或清理半成品后重采

可以一次性采多段，也可以一条一条采。当前默认推荐逐条采集：

1. 第一次填写新 `repo_id`，`采集段数` 设为 `1`，不要勾选 `继续写入已有数据集`。
2. 后续继续使用同一个 `repo_id`，勾选 `继续写入已有数据集`，`采集段数` 仍设为 `1`。
3. 每次完成后都会追加新的 episode 到同一个数据集。

逐条采集更适合早期调试，因为每条都能检查质量；流程稳定后可把 `采集段数` 改大批量采集。

数据检查可参考：

```bash
python examples/alohamini/replay_bi.py \
  --dataset local/alohamini_pick_lift_move_test_01 \
  --episode 0 \
  --remote_ip <Pi_IP> \
  --robot_model alohamini2pro

lerobot-dataset-viz \
  --repo-id "$ALOHAMINI_DATASET_HOME/local/alohamini_pick_lift_move_test_01" \
  --episode-index 0 \
  --display-compressed-images
```

旧的批量采集方式仍可用：

1. 打开 `数据` 页，填写：
   - `数据集 repo_id`: 例如 `local/alohamini_pick_lift_move_30`
   - `采集段数`: 例如 `30`
   - `FPS`: 通常 `25` 或按现场稳定帧率调整
   - `每段时长秒`: 单段任务最大时长
   - `重置时长秒`: 两段之间整理环境的时间
   - `任务描述`: 训练时使用的 task 文本
2. 点击 `开始数据采集`。

数据集默认单独保存到：

```bash
$ALOHAMINI_DATASET_HOME
```

也就是通过 `HF_LEROBOT_HOME` 覆盖 LeRobot 默认 cache。实际数据集路径为：

```bash
$ALOHAMINI_DATASET_HOME/<repo_id>
```

例如 GUI 中数据集填写 `local/alohamini_pick_lift_move_30` 时，本地保存路径是：

```bash
$ALOHAMINI_DATASET_HOME/local/alohamini_pick_lift_move_30
```

Leader 校准文件不放在数据集目录里，仍使用：

```bash
$ALOHAMINI_CALIBRATION_HOME
```

采集脚本会通过 `HF_LEROBOT_CALIBRATION` 指向该目录，并在启动前检查 `${LEADER_ID}_left.json` 和 `${LEADER_ID}_right.json` 是否存在。否则后台采集会停在 Leader 校准提示，无法进入 WASD 控制和数据记录循环。

采集前保持树莓派 Host 已启动；不要同时运行 GUI 遥操或单独相机预览，避免多个客户端同时占用同一条 ZMQ observation/action 链路。

旧入口仍可用，会转到新 GUI：

```bash
python alohamini_ops/alohamini_gui.py
```

## 校准

启动 Leader 校准：

```bash
alohamini_ops/calibrate_leaders.sh
```

自动写入已有 Leader 校准：

```bash
alohamini_ops/use_leader_calibration.sh
```

校准只能脚本化命令入口，不能脚本化物理过程。重新校准时必须在提示对应手臂时移动对应手臂，否则会再次出现左右映射或范围不可信的问题。

## 减少刷屏

`start_teleop.sh` 会把完整输出保存到 `/tmp/alohamini_teleop.log`。新版终端按键遥操只在状态变化或约 1 秒一次时打印紧凑状态，例如：

```text
[keys=forward] x= 0.250 y= 0.000 theta= 0.0 lift_vel=0 lift_h= 0.0
```

如果不需要 Rerun 可视化，启动时加 `--no_rerun`，可以减少内存和 CPU 压力：

```bash
alohamini_ops/start_teleop.sh --no_rerun
```

## 串口和停止延迟

上一次“发送 stop 后没有立刻停下”更像是命令链路和串口读写阻塞叠加：

- base 电机实测能收到速度命令，编码器位置也在变化，说明底盘硬件和供电不是完全失效。
- 调试脚本曾出现 `device reports readiness to read but returned no data`，这通常发生在串口被多进程同时访问、设备瞬断、USB 抖动或总线读写超时时。
- host 有 1000 ms watchdog，收不到新命令时会 stop，所以最坏情况下会有约 1 秒级延迟。
- 现场应避免 host、`motors.py`、`wheels.py` 同时访问 `/dev/am_arm_follower_left`。
