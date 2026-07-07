from datetime import datetime
from pathlib import Path

from qt_compat import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QTextCursor,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]


class LogPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.records: list[tuple[str, str]] = []
        self.auto_log_path = Path(__file__).resolve().parents[1] / "alohamini_gui.log"
        self._write_auto_log("")
        self._write_auto_log(f"===== GUI session started {datetime.now().isoformat(timespec='seconds')} =====")
        self.level_filter = QComboBox()
        self.level_filter.addItems(LEVELS)
        self.level_filter.setCurrentText("INFO")
        self.pause_scroll = QCheckBox("暂停滚动")
        self.clear_button = QPushButton("清空")
        self.save_button = QPushButton("保存")
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(80)
        self.setMinimumHeight(110)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.level_filter)
        toolbar.addWidget(self.pause_scroll)
        toolbar.addStretch(1)
        toolbar.addWidget(self.clear_button)
        toolbar.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.text)

        self.level_filter.currentTextChanged.connect(self.render)
        self.clear_button.clicked.connect(self.clear)
        self.save_button.clicked.connect(self.save)

    def append(self, level: str, message: str) -> None:
        level = level if level in LEVELS else "INFO"
        stamp = datetime.now().strftime("%H:%M:%S")
        self.records.append((level, f"{stamp} [{level}] {message}"))
        self._write_auto_log(self.records[-1][1])
        if self._visible(level):
            self.text.append(self.records[-1][1])
            if not self.pause_scroll.isChecked():
                self.text.moveCursor(QTextCursor.MoveOperation.End)

    def clear(self) -> None:
        self.records.clear()
        self.text.clear()

    def save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Log", "alohamini_gui.log", "Log Files (*.log);;Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(line for _, line in self.records))

    def render(self) -> None:
        self.text.clear()
        for level, line in self.records:
            if self._visible(level):
                self.text.append(line)

    def _visible(self, level: str) -> bool:
        return LEVELS.index(level) >= LEVELS.index(self.level_filter.currentText())

    def _write_auto_log(self, line: str) -> None:
        try:
            with open(self.auto_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
