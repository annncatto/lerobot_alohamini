from qt_compat import QLabel, QPushButton, QVBoxLayout, QWidget


class CalibrationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.calibrate = QPushButton("打开主臂校准终端")
        self.use_existing = QPushButton("写入已有主臂校准")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("校准仍是人工引导流程。提示哪条臂，就只移动哪条臂。"))
        layout.addWidget(self.calibrate)
        layout.addWidget(self.use_existing)
        layout.addStretch(1)
