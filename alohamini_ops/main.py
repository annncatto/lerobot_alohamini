#!/usr/bin/env python3
import signal
import sys

from app.context import build_context
from qt_compat import QApplication, QTimer
from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    sigint_timer = QTimer()
    sigint_timer.start(200)
    sigint_timer.timeout.connect(lambda: None)
    window = MainWindow(build_context())
    signal.signal(signal.SIGINT, lambda *_args: window.close())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
