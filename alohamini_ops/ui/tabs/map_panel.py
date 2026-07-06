import math
import time

from qt_compat import QColor, QHBoxLayout, QLabel, QPainter, QPen, QPushButton, QVBoxLayout, QWidget, Qt


class MapCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 360)
        self._points: list[tuple[float, float]] = [(0.0, 0.0)]
        self._pose = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._points = [(0.0, 0.0)]
        self._pose = (0.0, 0.0, 0.0)
        self.update()

    def set_pose(self, x: float, y: float, theta: float) -> None:
        self._pose = (x, y, theta)
        if not self._points or math.hypot(x - self._points[-1][0], y - self._points[-1][1]) > 0.002:
            self._points.append((x, y))
            self._points = self._points[-1200:]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#10151a"))

        margin = 28
        w = max(self.width() - 2 * margin, 1)
        h = max(self.height() - 2 * margin, 1)
        cx = margin + w / 2
        cy = margin + h / 2
        scale = self._scale_for_view(w, h)

        grid_pen = QPen(QColor("#26313a"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        for i in range(-5, 6):
            x = cx + i * scale
            y = cy + i * scale
            painter.drawLine(int(x), margin, int(x), margin + h)
            painter.drawLine(margin, int(y), margin + w, int(y))

        axis_pen = QPen(QColor("#4b5965"))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)
        painter.drawLine(margin, int(cy), margin + w, int(cy))
        painter.drawLine(int(cx), margin, int(cx), margin + h)

        if len(self._points) > 1:
            trace_pen = QPen(QColor("#61d394"))
            trace_pen.setWidth(3)
            painter.setPen(trace_pen)
            for a, b in zip(self._points, self._points[1:]):
                ax, ay = self._to_screen(a[0], a[1], cx, cy, scale)
                bx, by = self._to_screen(b[0], b[1], cx, cy, scale)
                painter.drawLine(int(ax), int(ay), int(bx), int(by))

        x, y, theta = self._pose
        rx, ry = self._to_screen(x, y, cx, cy, scale)
        robot_pen = QPen(QColor("#f2c14e"))
        robot_pen.setWidth(3)
        painter.setPen(robot_pen)
        painter.drawEllipse(int(rx - 7), int(ry - 7), 14, 14)
        heading_len = 26
        hx = rx + heading_len * math.cos(theta)
        hy = ry - heading_len * math.sin(theta)
        painter.drawLine(int(rx), int(ry), int(hx), int(hy))

        painter.setPen(QColor("#90a0ad"))
        painter.drawText(12, self.height() - 12, "估计轨迹，仅用于观察")

    def _scale_for_view(self, w: float, h: float) -> float:
        max_abs = 0.5
        for x, y in self._points:
            max_abs = max(max_abs, abs(x), abs(y))
        return min(w, h) / (2.2 * max_abs)

    @staticmethod
    def _to_screen(x: float, y: float, cx: float, cy: float, scale: float) -> tuple[float, float]:
        return cx + x * scale, cy - y * scale


class MapPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.canvas = MapCanvas()
        self.summary = QLabel("x=0.000 m  y=0.000 m  theta=0.0 deg")
        self.reset_button = QPushButton("清空轨迹")
        self.reset_button.clicked.connect(self.reset)
        self._last_t: float | None = None
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.summary, 1)
        toolbar.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.canvas, 1)

    def reset(self) -> None:
        self._last_t = None
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self.canvas.reset()
        self.summary.setText("x=0.000 m  y=0.000 m  theta=0.0 deg")

    def update_observation(self, observation: dict) -> None:
        now = time.monotonic()
        if self._last_t is None:
            self._last_t = now
            return
        dt = min(max(now - self._last_t, 0.0), 0.2)
        self._last_t = now

        vx = float(observation.get("x.vel", 0.0) or 0.0)
        vy = float(observation.get("y.vel", 0.0) or 0.0)
        omega = math.radians(float(observation.get("theta.vel", 0.0) or 0.0))

        cos_t = math.cos(self._theta)
        sin_t = math.sin(self._theta)
        self._x += (vx * cos_t - vy * sin_t) * dt
        self._y += (vx * sin_t + vy * cos_t) * dt
        self._theta += omega * dt
        self._theta = math.atan2(math.sin(self._theta), math.cos(self._theta))

        self.canvas.set_pose(self._x, self._y, self._theta)
        self.summary.setText(
            f"x={self._x: .3f} m  y={self._y: .3f} m  theta={math.degrees(self._theta): .1f} deg"
        )
