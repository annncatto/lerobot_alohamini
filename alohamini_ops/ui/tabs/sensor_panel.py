from qt_compat import QGridLayout, QLabel, QTextEdit, QVBoxLayout, QWidget


class SensorPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.base = QLabel("x=--  y=--  theta=--")
        self.lift = QLabel("height=--")
        self.action = QLabel("最近动作: --")
        self.left_arm = QTextEdit()
        self.right_arm = QTextEdit()
        self.other = QTextEdit()
        for box in (self.left_arm, self.right_arm, self.other):
            box.setReadOnly(True)
            box.setMinimumHeight(120)

        grid = QGridLayout()
        grid.addWidget(QLabel("底盘速度"), 0, 0)
        grid.addWidget(self.base, 0, 1)
        grid.addWidget(QLabel("升降轴"), 1, 0)
        grid.addWidget(self.lift, 1, 1)
        grid.addWidget(QLabel("动作"), 2, 0)
        grid.addWidget(self.action, 2, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(QLabel("左臂关节"))
        layout.addWidget(self.left_arm)
        layout.addWidget(QLabel("右臂关节"))
        layout.addWidget(self.right_arm)
        layout.addWidget(QLabel("其他状态"))
        layout.addWidget(self.other)

    def update_action(self, action: dict) -> None:
        self.action.setText(
            "最近动作: "
            f"x={float(action.get('x.vel', 0.0)): .3f}  "
            f"y={float(action.get('y.vel', 0.0)): .3f}  "
            f"theta={float(action.get('theta.vel', 0.0)): .1f}  "
            f"lift={action.get('lift_axis.vel', 0)}"
        )

    def update_observation(self, observation: dict) -> None:
        self.base.setText(
            f"x={self._fmt(observation.get('x.vel'))}  "
            f"y={self._fmt(observation.get('y.vel'))}  "
            f"theta={self._fmt(observation.get('theta.vel'))}"
        )
        self.lift.setText(f"height={self._fmt(observation.get('lift_axis.height_mm'))} mm")

        left = {}
        right = {}
        other = {}
        for key, value in observation.items():
            if self._is_image(value) or key == "observation.state":
                continue
            if key.startswith("arm_left_"):
                left[key] = value
            elif key.startswith("arm_right_"):
                right[key] = value
            elif key not in {"x.vel", "y.vel", "theta.vel", "lift_axis.height_mm"}:
                other[key] = value

        self.left_arm.setPlainText(self._format_group(left))
        self.right_arm.setPlainText(self._format_group(right))
        self.other.setPlainText(self._format_group(other))

    @staticmethod
    def _fmt(value) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value): .3f}"
        except Exception:
            return str(value)

    @staticmethod
    def _is_image(value) -> bool:
        return hasattr(value, "shape") and len(value.shape) == 3

    def _format_group(self, values: dict) -> str:
        if not values:
            return "--"
        lines = []
        for key in sorted(values):
            lines.append(f"{key}: {self._fmt(values[key])}")
        return "\n".join(lines)
