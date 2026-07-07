# AlohaMini 本机 GUI 使用

## 本机推荐入口

你当前电脑上推荐继续使用外层 GUI：

```bash
cd /home/anncatto/Desktop/Alohamini
alohamini_ops/start_gui.sh
```

这份外层 GUI 的配置文件是：

```bash
/home/anncatto/Desktop/Alohamini/alohamini_ops/config.env
```

它已经指向当前本机源码：

```bash
LOCAL_REPO=/home/anncatto/Desktop/Alohamini/lerobot_alohamini
PI_USER=pi5
PI_HOST=192.168.0.24
```

## 客户仓库入口

客户或新电脑 clone 后使用仓库内 GUI：

```bash
git clone git@github.com:annncatto/lerobot_alohamini.git
cd lerobot_alohamini
alohamini_ops/init_customer_env.sh --pi-user pi5 --pi-host <树莓派IP>
alohamini_ops/start_gui.sh
```

仓库内 GUI 的配置文件是：

```bash
lerobot_alohamini/alohamini_ops/config.env
```

客户首次运行 `init_customer_env.sh` 后，会自动把本机路径、conda 路径、数据集路径和树莓派地址写进去。

`setup_env.sh` 只在新电脑首次安装、Qt/依赖缺失或环境损坏时运行：

```bash
alohamini_ops/setup_env.sh
```

如果需要启用麦克风语音控制，再安装可选语音依赖：

```bash
alohamini_ops/setup_env.sh --voice
```

## GUI 常用顺序

1. 打开 GUI：`alohamini_ops/start_gui.sh`
2. 在 `连接` 页确认或修改 `PI_USER`、`PI_HOST`。
3. 点击 `启动 Host`。
4. 点击 `刷新状态`，确认树莓派设备和 host 进程正常。
5. 在 `诊断` 页先用：
   - `调试机械臂串口号`
   - `检查本机 Leader 舵机`
   - `检查树莓派 Follower/底盘舵机`
   - `检查升降轴`
6. 遥操作用 `遥操` 页。
7. 数据采集用 `数据` 页。
8. 训练后真机 rollout 用 `部署` 页。

## 标准相机打开流程

相机列表由树莓派 Host 启动时决定。GUI 只能显示 Host 已经打开的相机；如果 Host 已经用两路相机启动，后面在 GUI 勾选五路不会热切换。

推荐方式是在 GUI 右侧 `相机` 页勾选相机后点击 `保存相机配置`。它会把列表写入：

```bash
alohamini_ops/config.env
```

对应字段：

```bash
ALOHAMINI_CAMERAS=forward,wrist_right
```

`src/lerobot/robots/alohamini/config_lekiwi.py` 里保留 5 路相机目录，不建议客户通过删除/取消注释源码来切换相机。

终端启用其它相机：

```bash
cd /home/anncatto/Desktop/Alohamini/lerobot_alohamini

# 修改 alohamini_ops/config.env，例：开启 5 路
sed -i 's/^ALOHAMINI_CAMERAS=.*/ALOHAMINI_CAMERAS=forward,backward,chest,wrist_left,wrist_right/' alohamini_ops/config.env

# 重启树莓派 Host，让新相机列表生效
alohamini_ops/stop_pi_host.sh
alohamini_ops/start_pi_host.sh
```

如果只想开三路，例如前视、胸前、右腕：

```bash
sed -i 's/^ALOHAMINI_CAMERAS=.*/ALOHAMINI_CAMERAS=forward,chest,wrist_right/' alohamini_ops/config.env
alohamini_ops/stop_pi_host.sh
alohamini_ops/start_pi_host.sh
```

注意上面的 `ALOHAMINI_CAMERAS=...` 需要写入 `alohamini_ops/config.env`，因为脚本会读取这个文件。只在命令前临时加环境变量会被 `config.env` 覆盖。

确认树莓派相机设备存在：

```bash
ssh pi5@192.168.0.24 'ls -l /dev/am_camera_* /dev/video* 2>/dev/null || true'
```

GUI 接入：

1. 在 `连接` 页确认 `PI_HOST/PI_USER` 正确。
2. 如果改了相机列表，先点击 `停止 Host`，再点击 `启动 Host`。
3. 点击 `刷新状态`，确认 Host 正常。
4. 打开右侧 `相机` 页。
5. 勾选和 Host 一致的相机，默认建议先用 `forward` 和 `wrist_right`。
6. 点击 `打开已选相机` 或 `应用并打开`。
7. 看到画面后再启动遥操；如果已经在 GUI 遥操中，右侧相机页会复用同一个遥操客户端的 observation。

注意：

- `应用并打开` 会应用勾选列表并启动预览，不只是保存配置。
- 数据采集运行中不要单独打开相机预览，采集进程会自己接入相机画面。
- 如果勾选了未插入或未映射的相机，Host/相机预览可能报错；先用两路默认相机确认链路。

## 串口调试怎么看

`调试机械臂串口号` 只读检查，不会改系统。重点看：

- `/dev/am_arm_leader_left/right` 是否存在，是否指向正确 `/dev/ttyACM*`。
- `/dev/am_arm_follower_left/right` 是否存在，是否指向树莓派上的正确 `/dev/ttyACM*`。
- `/dev/serial/by-id/*` 的真实路径。
- 每个候选串口扫描到的舵机 ID。

预期分布：

- 本机 Leader left/right：通常各自扫描到 ID `1-7`。
- 树莓派 follower left：通常有左臂 `1-7`、底盘 `8/9/10`、升降 `11`。
- 树莓派 follower right：通常有右臂 `1-7`。

如果映射不对，根据 by-id、真实路径和舵机 ID 分布人工修 udev 或重新插线确认。

## 两份 GUI 怎么同步

开发时以仓库内为准：

```bash
/home/anncatto/Desktop/Alohamini/lerobot_alohamini/alohamini_ops
```

如果改了仓库内 GUI，又想立刻在当前电脑外层入口使用，同步：

```bash
rsync -a --delete \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.log' \
  --exclude='record_control.txt' \
  --exclude='record_motion.json' \
  /home/anncatto/Desktop/Alohamini/lerobot_alohamini/alohamini_ops/ \
  /home/anncatto/Desktop/Alohamini/alohamini_ops/
```

同步后仍使用：

```bash
cd /home/anncatto/Desktop/Alohamini
alohamini_ops/start_gui.sh
```
