#!/usr/bin/env python3
import sys

from app.context import build_context
from qt_compat import QApplication
from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(build_context())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
