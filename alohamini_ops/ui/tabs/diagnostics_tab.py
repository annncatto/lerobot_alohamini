from qt_compat import QLabel, QPushButton, QVBoxLayout, QWidget


class DiagnosticsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.status = QPushButton("运行状态检查")
        self.host_log = QPushButton("查看 Host 日志")
        self.local_log = QPushButton("查看本机遥操日志")
        self.local_servos = QPushButton("检查本机 Leader 舵机")
        self.pi_servos = QPushButton("检查树莓派 Follower/底盘舵机")
        self.serial_ports = QPushButton("串口定位/舵机扫描")
        self.lift_axis = QPushButton("检查升降轴")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("诊断"))
        self.status.setToolTip("检查本机 Leader、树莓派 Follower、Host 进程和 Host 日志。")
        self.serial_ports.setToolTip("定位 /dev/ttyACM* 对应哪条机械臂，并扫描舵机 ID。")
        self.lift_axis.setToolTip("只检查升降轴方向、反馈和运动。")
        self.host_log.setToolTip("查看树莓派 Host 最近日志。")
        self.local_log.setToolTip("查看本机遥操最近日志。")
        layout.addWidget(self.status)
        layout.addWidget(self.serial_ports)
        layout.addWidget(self.lift_axis)
        layout.addWidget(self.host_log)
        layout.addWidget(self.local_log)
        layout.addStretch(1)
