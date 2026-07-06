from qt_compat import QLabel, QPushButton, QVBoxLayout, QWidget


class DiagnosticsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.status = QPushButton("运行状态检查")
        self.host_log = QPushButton("查看 Host 日志")
        self.local_log = QPushButton("查看本机遥操日志")
        self.local_servos = QPushButton("检查本机 Leader 舵机")
        self.pi_servos = QPushButton("检查树莓派 Follower/底盘舵机")
        self.lift_axis = QPushButton("检查升降轴")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("诊断"))
        layout.addWidget(self.status)
        layout.addWidget(self.local_servos)
        layout.addWidget(self.pi_servos)
        layout.addWidget(self.lift_axis)
        layout.addWidget(self.host_log)
        layout.addWidget(self.local_log)
        layout.addStretch(1)
