import time
from pathlib import Path

from qt_compat import QObject, QImage, Signal, Slot


class RecordPreviewWorker(QObject):
    frames = Signal(object)
    log = Signal(str, str)
    finished = Signal()

    def __init__(self, preview_dir: Path, fps: int = 8):
        super().__init__()
        self.preview_dir = Path(preview_dir)
        self.fps = max(int(fps), 1)
        self.running = True
        self._last_mtime_by_name: dict[str, float] = {}

    @Slot()
    def run(self) -> None:
        self.log.emit("INFO", f"采集相机预览已接入 GUI: {self.preview_dir}")
        delay = 1.0 / self.fps
        while self.running:
            images = {}
            for path in sorted(self.preview_dir.glob("*.jpg")):
                mtime = path.stat().st_mtime
                if self._last_mtime_by_name.get(path.stem) == mtime:
                    continue
                image = QImage(str(path))
                if image.isNull():
                    continue
                self._last_mtime_by_name[path.stem] = mtime
                images[path.stem] = image
            if images:
                self.frames.emit(images)
            time.sleep(delay)
        self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self.running = False
