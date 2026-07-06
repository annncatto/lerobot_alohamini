import time
from threading import Lock

from qt_compat import QObject, QImage, Signal, Slot


class TeleopWorker(QObject):
    log = Signal(str, str)
    state = Signal(dict)
    observation = Signal(dict)
    connected = Signal(bool)
    frame = Signal(object)
    frames = Signal(object)
    sources = Signal(object)
    finished = Signal()

    def __init__(self, config: dict, use_leader: bool, camera_name: str = "auto"):
        super().__init__()
        self.config = config
        self.use_leader = use_leader
        self.running = True
        self.estop = False
        self._lock = Lock()
        self._base = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        self._lift = {"lift_axis.vel": 0}
        self._base_deadline = 0.0
        self._lift_deadline = 0.0
        self._keyboard_keys: set[str] = set()
        self._keyboard_deadline = 0.0
        self._command_ttl = 0.35
        self._speed = 1.0
        self._camera_name = camera_name or "auto"
        self._camera_enabled = False
        self._camera_emit_period = 1.0 / 10
        self._last_camera_emit = 0.0
        self.robot = None
        self.leader = None

    @Slot()
    def run(self) -> None:
        try:
            import cv2
            from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
            from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
            from lerobot.teleoperators.so_leader import SOLeaderConfig
            from lerobot.utils.robot_utils import precise_sleep

            robot_cfg = LeKiwiClientConfig(
                remote_ip=self.config["pi_host"],
                id="my_alohamini",
                robot_model=self.config["robot_model"],
            )
            self.robot = LeKiwiClient(robot_cfg)
            if self.robot.config.cameras:
                self.sources.emit(list(self.robot.config.cameras.keys()))
            self.robot.connect()

            if self.use_leader:
                bi_cfg = BiSOLeaderConfig(
                    left_arm_config=SOLeaderConfig(
                        port="/dev/am_arm_leader_left",
                        arm_profile=self.config["arm_profile"],
                    ),
                    right_arm_config=SOLeaderConfig(
                        port="/dev/am_arm_leader_right",
                        arm_profile=self.config["arm_profile"],
                    ),
                    id=self.config["leader_id"],
                )
                self.leader = BiSOLeader(bi_cfg)
                self.leader.connect()

            self.connected.emit(True)
            self.log.emit("INFO", "GUI teleop connected.")

            while self.running:
                t0 = time.perf_counter()
                action = {}
                observation = self.robot.get_observation()
                self.observation.emit(observation)
                self._emit_camera_frames(observation, cv2)
                if self.leader is not None:
                    action.update({f"arm_{k}": v for k, v in self.leader.get_action().items()})
                with self._lock:
                    now = time.monotonic()
                    if self.estop:
                        base = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                        lift = {"lift_axis.vel": 0}
                    elif self._keyboard_keys and now <= self._keyboard_deadline:
                        keyboard_keys = dict.fromkeys(self._keyboard_keys, None)
                        base = self.robot._from_keyboard_to_base_action(keyboard_keys)
                        lift = self.robot._from_keyboard_to_lift_action(keyboard_keys)
                    else:
                        base = dict(self._base) if now <= self._base_deadline else {
                            "x.vel": 0.0,
                            "y.vel": 0.0,
                            "theta.vel": 0.0,
                        }
                        lift = dict(self._lift) if now <= self._lift_deadline else {"lift_axis.vel": 0}
                        speed = self._speed
                        base["x.vel"] *= speed
                        base["y.vel"] *= speed
                        base["theta.vel"] *= speed
                action.update(base)
                action.update(lift)
                self.robot.send_action(action)
                self.state.emit(action)
                precise_sleep(max(1 / 30 - (time.perf_counter() - t0), 0.0))

        except Exception as exc:
            self.log.emit("ERROR", f"GUI teleop failed: {exc}")
            self.connected.emit(False)
        finally:
            try:
                if self.robot is not None:
                    self.robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.vel": 0})
            except Exception as exc:
                self.log.emit("WARN", f"Stop action failed: {exc}")
            try:
                if self.leader is not None:
                    self.leader.disconnect()
                if self.robot is not None:
                    self.robot.disconnect()
            except Exception as exc:
                self.log.emit("WARN", f"Disconnect failed: {exc}")
            self.leader = None
            self.robot = None
            self.connected.emit(False)
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self.running = False
        self.set_estop(True)

    def set_base(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        with self._lock:
            self._base = {"x.vel": x, "y.vel": y, "theta.vel": theta}
            self._base_deadline = time.monotonic() + self._command_ttl if any((x, y, theta)) else 0.0

    def set_lift(self, vel: int = 0) -> None:
        with self._lock:
            self._lift = {"lift_axis.vel": vel}
            self._lift_deadline = time.monotonic() + self._command_ttl if vel else 0.0

    def set_keyboard_keys(self, keys: set[str]) -> None:
        with self._lock:
            self._keyboard_keys = set(keys)
            self._keyboard_deadline = time.monotonic() + self._command_ttl if keys else 0.0

    def set_speed_scale(self, speed: float) -> None:
        with self._lock:
            self._speed = speed

    def set_camera_name(self, camera_name: str) -> None:
        with self._lock:
            self._camera_name = camera_name or "auto"

    def set_camera_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._camera_enabled = enabled
            if enabled:
                self._last_camera_emit = 0.0

    def set_estop(self, enabled: bool) -> None:
        with self._lock:
            self.estop = enabled
            if enabled:
                self._base = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                self._lift = {"lift_axis.vel": 0}
                self._base_deadline = 0.0
                self._lift_deadline = 0.0
                self._keyboard_keys.clear()
                self._keyboard_deadline = 0.0

    def _emit_camera_frames(self, observation: dict, cv2) -> None:
        with self._lock:
            enabled = self._camera_enabled
            camera_name = self._camera_name
        if not enabled:
            return

        now = time.monotonic()
        if now - self._last_camera_emit < self._camera_emit_period:
            return

        selected = self._select_frames(observation, camera_name)
        if not selected:
            return

        images = {}
        for name, frame in selected.items():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            images[name] = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.frames.emit(images)
        first = next(iter(images.values()), None)
        if first is not None:
            self.frame.emit(first)
        self._last_camera_emit = now

    def _select_frames(self, observation: dict, camera_name: str) -> dict:
        if camera_name == "auto":
            frames = {}
            for name, value in observation.items():
                if hasattr(value, "shape") and len(value.shape) == 3:
                    frames[name] = value
            return frames
        value = observation.get(camera_name)
        if hasattr(value, "shape") and len(value.shape) == 3:
            return {camera_name: value}
        return {}

    def _select_frame(self, observation: dict, camera_name: str):
        selected = self._select_frames(observation, camera_name)
        if selected:
            return next(iter(selected.values()))
        if camera_name == "auto":
            for value in observation.values():
                if hasattr(value, "shape") and len(value.shape) == 3:
                    return value
            return None
        value = observation.get(camera_name)
        if hasattr(value, "shape") and len(value.shape) == 3:
            return value
        return None
