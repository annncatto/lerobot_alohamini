from qt_compat import QAction, QKeySequence


class RobotActions:
    def __init__(self, window):
        self.window = window
        self.actions: list[QAction] = []

    def install(self) -> None:
        bindings = [
            ("Forward", "W", lambda: self.window.set_base(0.25, 0.0, 0.0)),
            ("Backward", "S", lambda: self.window.set_base(-0.25, 0.0, 0.0)),
            ("Left", "Z", lambda: self.window.set_base(0.0, 0.25, 0.0)),
            ("Right", "X", lambda: self.window.set_base(0.0, -0.25, 0.0)),
            ("Rotate Left", "A", lambda: self.window.set_base(0.0, 0.0, 60.0)),
            ("Rotate Right", "D", lambda: self.window.set_base(0.0, 0.0, -60.0)),
            ("Lift Up", "U", lambda: self.window.set_lift(1000)),
            ("Lift Down", "J", lambda: self.window.set_lift(-1000)),
            ("Stop", "Space", self.window.stop_motion),
            ("EStop", "Esc", self.window.emergency_stop),
        ]
        for name, key, callback in bindings:
            action = QAction(name, self.window)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            self.window.addAction(action)
            self.actions.append(action)
