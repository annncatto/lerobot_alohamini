import math
import os
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from qt_compat import QObject, Signal, Slot
except Exception:  # pragma: no cover - terminal teleop can use this module without Qt.
    QObject = None
    Signal = None
    Slot = lambda *args, **kwargs: (lambda fn: fn)


MOTION_KEY_BY_COMMAND = {
    "forward": "w",
    "backward": "s",
    "left": "z",
    "right": "x",
    "rotate_left": "a",
    "rotate_right": "d",
    "lift_up": "u",
    "lift_down": "j",
}

VOICE_XY_SPEED = 0.05
VOICE_THETA_SPEED = 15.0
VOICE_LIFT_VEL = 300


@dataclass(frozen=True)
class VoiceCommand:
    kind: str
    name: str
    text: str
    key: str | None = None

    def as_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "text": self.text, "key": self.key}


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def parse_voice_command(text: str) -> VoiceCommand | None:
    normalized = text.lower().replace(" ", "").replace("，", ",").replace("。", ".").strip()
    if not normalized:
        return None

    # 录制控制要优先匹配，避免“停止录制”被当成遥操急停。
    if _contains_any(normalized, ("保存", "结束本段", "完成当前段", "finishepisode", "saveepisode")):
        return VoiceCommand("record", "finish_wait", text)
    if _contains_any(normalized, ("废弃", "重录", "丢弃", "rerecord", "discard")):
        return VoiceCommand("record", "rerecord_wait", text)
    if _contains_any(normalized, ("继续采集", "继续录制", "复位完成", "重新开始", "restart", "resume")):
        return VoiceCommand("record", "restart", text)
    if _contains_any(normalized, ("停止录制", "停止采集", "stoprecord", "stoprecording")):
        return VoiceCommand("record", "stop", text)

    if _contains_any(normalized, ("急停", "停止", "停下", "stop", "estop", "emergencystop")):
        return VoiceCommand("emergency_stop", "emergency_stop", text)
    if _contains_any(normalized, ("前进", "向前", "forward")):
        return VoiceCommand("motion", "forward", text, MOTION_KEY_BY_COMMAND["forward"])
    if _contains_any(normalized, ("后退", "向后", "backward", "back")):
        return VoiceCommand("motion", "backward", text, MOTION_KEY_BY_COMMAND["backward"])
    if _contains_any(normalized, ("左转", "向左转", "rotateleft", "turnleft")):
        return VoiceCommand("motion", "rotate_left", text, MOTION_KEY_BY_COMMAND["rotate_left"])
    if _contains_any(normalized, ("右转", "向右转", "rotateright", "turnright")):
        return VoiceCommand("motion", "rotate_right", text, MOTION_KEY_BY_COMMAND["rotate_right"])
    if _contains_any(normalized, ("左移", "向左平移", "moveleft", "strafeleft")):
        return VoiceCommand("motion", "left", text, MOTION_KEY_BY_COMMAND["left"])
    if _contains_any(normalized, ("右移", "向右平移", "moveright", "straferight")):
        return VoiceCommand("motion", "right", text, MOTION_KEY_BY_COMMAND["right"])
    if _contains_any(normalized, ("上升", "升高", "liftup", "up")):
        return VoiceCommand("motion", "lift_up", text, MOTION_KEY_BY_COMMAND["lift_up"])
    if _contains_any(normalized, ("下降", "降低", "liftdown", "down")):
        return VoiceCommand("motion", "lift_down", text, MOTION_KEY_BY_COMMAND["lift_down"])
    return None


def low_speed_motion_action(command_name: str) -> dict:
    """Return a low-speed continuous action for voice teleop.

    Keep these speed constants in the ops voice layer so voice control does not
    change keyboard speed_index behavior inside LeKiwiClient.
    """
    if command_name == "forward":
        return {"x.vel": VOICE_XY_SPEED, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.vel": 0}
    if command_name == "backward":
        return {"x.vel": -VOICE_XY_SPEED, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.vel": 0}
    if command_name == "left":
        return {"x.vel": 0.0, "y.vel": VOICE_XY_SPEED, "theta.vel": 0.0, "lift_axis.vel": 0}
    if command_name == "right":
        return {"x.vel": 0.0, "y.vel": -VOICE_XY_SPEED, "theta.vel": 0.0, "lift_axis.vel": 0}
    if command_name == "rotate_left":
        return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": VOICE_THETA_SPEED, "lift_axis.vel": 0}
    if command_name == "rotate_right":
        return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -VOICE_THETA_SPEED, "lift_axis.vel": 0}
    if command_name == "lift_up":
        return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.vel": VOICE_LIFT_VEL}
    if command_name == "lift_down":
        return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.vel": -VOICE_LIFT_VEL}
    return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.vel": 0}


def pick_input_device(devices, requested: str | None = None) -> int | None:
    if requested is not None and requested.isdigit():
        idx = int(requested)
        if 0 <= idx < len(devices) and devices[idx].get("max_input_channels", 0) > 0:
            return idx
    try:
        import sounddevice as sd

        default_in = sd.default.device[0]
        if default_in is not None and devices[default_in].get("max_input_channels", 0) > 0:
            return default_in
    except Exception:
        pass
    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            return i
    return None


def _write_wav(path: str, audio: np.ndarray, samplerate: int) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(samplerate)
        wav.writeframes(audio.astype(np.int16).tobytes())


class VoiceRecognizer:
    def __init__(
        self,
        model_name: str = "small",
        language: str = "zh",
        device_index: str | None = None,
        sample_seconds: float = 1.8,
        samplerate: int = 16000,
        min_rms: float = 0.008,
    ):
        self.model_name = model_name
        self.language = language
        self.device_index = device_index
        self.sample_seconds = sample_seconds
        self.samplerate = samplerate
        self.min_rms = min_rms
        self._sd = None
        self._model = None
        self._input_device = None

    def setup(self, log: Callable[[str, str], None] | None = None) -> None:
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "语音控制依赖缺失，请先在 lerobot_alohamini 环境安装 sounddevice 和 faster-whisper。"
            ) from exc

        devices = sd.query_devices()
        selected = pick_input_device(devices, self.device_index or os.getenv("VOICE_DEVICE_INDEX"))
        if selected is None:
            raise RuntimeError("没有找到可用麦克风。可设置 VOICE_DEVICE_INDEX 指定输入设备。")
        self._sd = sd
        self._input_device = selected
        sd.default.device = (selected, None)
        if log is not None:
            log("INFO", f"语音控制使用麦克风设备 {selected}: {devices[selected]['name']}")
            log("INFO", f"正在加载 Whisper 模型: {self.model_name}")
        self._model = WhisperModel(self.model_name, device="cpu")

    def listen_once(self) -> tuple[str, VoiceCommand | None] | None:
        if self._sd is None or self._model is None or self._input_device is None:
            raise RuntimeError("VoiceRecognizer.setup() must be called before listen_once().")

        samples = int(self.sample_seconds * self.samplerate)
        audio = self._sd.rec(samples, samplerate=self.samplerate, channels=1, dtype="int16")
        self._sd.wait()
        mono = audio.reshape(-1)
        rms = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2)) + 1e-12)
        if rms < self.min_rms:
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            _write_wav(tmp_path, mono, self.samplerate)
            segments, _info = self._model.transcribe(tmp_path, language=self.language)
            text = "".join(segment.text for segment in segments).strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if not text:
            return None
        return text, parse_voice_command(text)

    def run_forever(
        self,
        stop_event: threading.Event,
        on_text: Callable[[str], None] | None = None,
        on_command: Callable[[VoiceCommand], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ) -> None:
        self.setup(on_log)
        if on_log is not None:
            on_log("INFO", "语音控制已启动。")
        while not stop_event.is_set():
            result = self.listen_once()
            if result is None:
                continue
            text, command = result
            if on_text is not None:
                on_text(text)
            if command is not None:
                if on_command is not None:
                    on_command(command)
            elif on_log is not None:
                on_log("INFO", f"未匹配语音命令: {text}")
            time.sleep(0.05)


class VoiceCommandThread(threading.Thread):
    def __init__(
        self,
        on_command: Callable[[dict], None],
        on_log: Callable[[str, str], None] | None = None,
        model_name: str = "small",
        language: str = "zh",
        device_index: str | None = None,
    ):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._recognizer = VoiceRecognizer(model_name=model_name, language=language, device_index=device_index)
        self._on_command = on_command
        self._on_log = on_log

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            self._recognizer.run_forever(
                self._stop_event,
                on_text=lambda text: self._log("INFO", f"识别到语音: {text}"),
                on_command=lambda command: self._on_command(command.as_dict()),
                on_log=self._log,
            )
        except Exception as exc:
            self._log("ERROR", f"语音控制退出: {exc}")

    def _log(self, level: str, message: str) -> None:
        if self._on_log is not None:
            self._on_log(level, message)


if QObject is not None:

    class VoiceWorker(QObject):
        log = Signal(str, str)
        heard = Signal(str)
        command = Signal(object)
        finished = Signal()

        def __init__(
            self,
            model_name: str = "small",
            language: str = "zh",
            device_index: str | None = None,
        ):
            super().__init__()
            self._stop_event = threading.Event()
            self._recognizer = VoiceRecognizer(
                model_name=model_name,
                language=language,
                device_index=device_index,
            )

        @Slot()
        def run(self) -> None:
            try:
                self._recognizer.run_forever(
                    self._stop_event,
                    on_text=self.heard.emit,
                    on_command=lambda command: self.command.emit(command.as_dict()),
                    on_log=self.log.emit,
                )
            except Exception as exc:
                self.log.emit("ERROR", f"语音控制失败: {exc}")
            finally:
                self.finished.emit()

        @Slot()
        def stop(self) -> None:
            self._stop_event.set()

else:
    VoiceWorker = None
