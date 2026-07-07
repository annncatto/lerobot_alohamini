from qt_compat import QCheckBox, QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


class ConnectionTab(QWidget):
    def __init__(self, context):
        super().__init__()
        self.pi_user = QLineEdit(context.config.pi_user)
        self.pi_host = QLineEdit(context.config.pi_host)
        self.apply_pi = QPushButton("应用连接配置")
        self.save_pi = QPushButton("保存到 config.env")
        self.model_info = QLabel(f"型号: {context.config.robot_model}")
        self.model_info.setWordWrap(True)
        self.use_leader = QCheckBox("连接主臂 Leader")
        self.use_leader.setChecked(True)
        self.start_host = QPushButton("启动树莓派 Host")
        self.stop_host = QPushButton("停止树莓派 Host")
        self.start_teleop = QPushButton("连接 GUI 遥操")
        self.stop_teleop = QPushButton("断开 GUI 遥操")
        self.status_check = QPushButton("刷新状态")
        self.tail_log = QPushButton("查看 Host 日志")

        box = QGroupBox("连接")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(16, 8, 16, 14)
        box_layout.setSpacing(8)

        target_grid = QGridLayout()
        target_grid.addWidget(QLabel("Pi 用户"), 0, 0)
        target_grid.addWidget(self.pi_user, 0, 1)
        target_grid.addWidget(QLabel("Pi 地址"), 1, 0)
        target_grid.addWidget(self.pi_host, 1, 1)

        box_layout.addWidget(self.model_info)
        box_layout.addLayout(target_grid)
        box_layout.addWidget(self.apply_pi)
        box_layout.addWidget(self.save_pi)
        for widget in [
            self.use_leader,
            self.start_host,
            self.stop_host,
            self.start_teleop,
            self.stop_teleop,
            self.status_check,
            self.tail_log,
        ]:
            box_layout.addWidget(widget)
        box_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addStretch(1)
