import math

from qt_compat import QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget, Qt


class CameraPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.source = QComboBox()
        self.source.addItems(["auto"])
        self.source.setEditable(False)
        self.connect = QPushButton("连接机器人相机")
        self.disconnect = QPushButton("停止机器人相机")
        self.capture = QPushButton("保存当前帧")
        self.frames: dict[str, QLabel] = {}
        self.frame_names: list[str] = []
        self._title_labels: dict[str, QLabel] = {}

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("机器人相机"))
        toolbar.addWidget(self.source, 1)
        toolbar.addWidget(self.connect)
        toolbar.addWidget(self.disconnect)
        toolbar.addWidget(self.capture)

        self.grid = QGridLayout()

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addLayout(self.grid, 1)
        self.set_frame_names([])

    def _make_frame(self, text: str) -> QLabel:
        frame = QLabel(text)
        frame.setObjectName("cameraFrame")
        frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame.setMinimumSize(320, 240)
        frame.setScaledContents(False)
        return frame

    def set_sources(self, names: list[str]) -> None:
        current = self.source.currentText()
        options = ["auto", *names]
        if [self.source.itemText(i) for i in range(self.source.count())] == options:
            self.set_frame_names(names)
            return
        self.source.clear()
        self.source.addItems(options)
        self.source.setCurrentText(current if current in options else "auto")
        self.set_frame_names(names)

    def set_frame_names(self, names: list[str]) -> None:
        visible_names = list(names) or ["相机预览"]
        if visible_names == self.frame_names:
            return
        self._clear_grid()
        self.frame_names = visible_names
        self.frames = {}
        self._title_labels = {}

        columns = 1 if len(visible_names) == 1 else 2
        rows = math.ceil(len(visible_names) / columns)
        for index, name in enumerate(visible_names):
            row = index // columns
            col = index % columns
            title = QLabel(name)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            frame = self._make_frame("未收到相机帧")
            self.grid.addWidget(title, row * 2, col)
            self.grid.addWidget(frame, row * 2 + 1, col)
            self.frames[name] = frame
            self._title_labels[name] = title
        for col in range(columns):
            self.grid.setColumnStretch(col, 1)
        for row in range(rows):
            self.grid.setRowStretch(row * 2 + 1, 1)

    def labels_for_images(self, names: list[str]) -> list[tuple[str, QLabel]]:
        if not names:
            return []
        if any(name not in self.frames for name in names):
            self.set_frame_names(names)
        return [(name, self.frames[name]) for name in names if name in self.frames]

    def first_label(self) -> QLabel | None:
        return next(iter(self.frames.values()), None)

    def clear_frames(self, text: str) -> None:
        for label in self.frames.values():
            label.clear()
            label.setText(text)

    def _clear_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
