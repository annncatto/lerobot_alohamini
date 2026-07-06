import time

from qt_compat import QObject, QImage, Signal, Slot


class CameraWorker(QObject):
    frame = Signal(object)
    frames = Signal(object)
    observation = Signal(dict)
    log = Signal(str, str)
    sources = Signal(object)
    finished = Signal()

    def __init__(self, config: dict, camera_name: str, fps: int = 20):
        super().__init__()
        self.config = config
        self.camera_name = camera_name
        self.fps = fps
        self.running = True
        self.robot = None

    @Slot()
    def run(self) -> None:
        try:
            import cv2
            from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig

            robot_cfg = LeKiwiClientConfig(
                remote_ip=self.config["pi_host"],
                id="my_alohamini_camera_viewer",
                robot_model=self.config["robot_model"],
            )
            self.robot = LeKiwiClient(robot_cfg)
            if not self.robot.config.cameras:
                self.log.emit("ERROR", "当前 lekiwi_cameras_config() 没有启用任何机器人摄像头。")
                return
            camera_names = list(self.robot.config.cameras.keys())
            configured = ", ".join(camera_names)
            self.sources.emit(camera_names)
            self.log.emit("INFO", f"机器人摄像头配置: {configured}")
            if self.camera_name != "auto" and self.camera_name not in camera_names:
                self.log.emit("ERROR", f"相机未配置: {self.camera_name}。可用相机: {configured}")
                return
            self.log.emit("WARN", "相机页会订阅 Host observation。不要同时运行另一个遥操客户端订阅同一 observation。")
            self.robot.connect()

            self.log.emit("INFO", f"机器人相机已连接: {self.camera_name}")
            delay = 1.0 / max(self.fps, 1)
            while self.running:
                obs = self.robot.get_observation()
                self.observation.emit(obs)
                selected = self._select_frames(obs)
                if not selected:
                    self.log.emit("WARN", f"未收到机器人相机帧: {self.camera_name}")
                    time.sleep(0.2)
                    continue

                images = {}
                for name, frame in selected.items():
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    images[name] = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                self.frames.emit(images)
                first = next(iter(images.values()), None)
                if first is not None:
                    self.frame.emit(first)
                time.sleep(delay)
        except Exception as exc:
            self.log.emit("ERROR", f"机器人相机线程异常: {exc}")
        finally:
            if self.robot is not None:
                try:
                    self.robot.disconnect()
                except Exception:
                    pass
                self.robot = None
            self.log.emit("INFO", "机器人相机已停止")
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self.running = False

    def _select_frame(self, obs: dict):
        selected = self._select_frames(obs)
        if selected:
            return next(iter(selected.values()))
        return None

    def _select_frames(self, obs: dict) -> dict:
        if self.camera_name == "auto":
            frames = {}
            for name, value in obs.items():
                if hasattr(value, "shape") and len(value.shape) == 3:
                    frames[name] = value
            return frames
        value = obs.get(self.camera_name)
        if hasattr(value, "shape") and len(value.shape) == 3:
            return {self.camera_name: value}
        return {}

    def _select_first_frame(self, obs: dict):
        if self.camera_name == "auto":
            for value in obs.values():
                if hasattr(value, "shape") and len(value.shape) == 3:
                    return value
            return None
        value = obs.get(self.camera_name)
        if hasattr(value, "shape") and len(value.shape) == 3:
            return value
        return None
