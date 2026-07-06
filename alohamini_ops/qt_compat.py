try:
    from PyQt6.QtCore import QPointF, QEvent, QObject, Qt, QThread, QTimer, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QTextCursor
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt6"
except ModuleNotFoundError:
    try:
        from PySide6.QtCore import QPointF, QEvent, QObject, Qt, QThread, QTimer, Signal, Slot
        from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QTextCursor
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSplitter,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        QT_API = "PySide6"
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Qt binding is missing. Install one of these in the lerobot_alohamini env:\n"
            "  pip install PyQt6\n"
            "or:\n"
            "  pip install PySide6"
        ) from exc
