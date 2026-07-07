import os
import re
import subprocess
import time
import json
from pathlib import Path

from app.context import AppContext
from qt_compat import (
    QApplication,
    QEvent,
    QFileDialog,
    QLineEdit,
    QMainWindow,
    QPixmap,
    QTextEdit,
    QThread,
    QTimer,
    Qt,
    Slot,
)
from ui.presenters.status_presenter import StatusPresenter
from ui.ui_manager import UiManager
from workers.command_worker import CommandWorker
from workers.camera_worker import CameraWorker
from workers.record_preview_worker import RecordPreviewWorker
from workers.teleop_worker import TeleopWorker
from workers.voice_worker import MOTION_KEY_BY_COMMAND, VoiceWorker, low_speed_motion_action


STYLE = """
QMainWindow, QWidget { background: #202428; color: #e7edf2; font-size: 13px; }
QTabWidget::pane, #visualPanel, #summaryPanel { border: 1px solid #3a424b; border-radius: 6px; }
QGroupBox { border: 1px solid #3a424b; border-radius: 6px; margin-top: 18px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px; background: #202428; }
QTabBar::tab { padding: 7px 12px; background: #2a3036; border: 1px solid #3a424b; }
QTabBar::tab:selected { background: #38536b; }
QPushButton { background: #34424d; border: 1px solid #4e5e69; border-radius: 5px; padding: 7px; }
QPushButton:hover { background: #40515f; }
QPushButton:pressed { background: #26343f; }
QPushButton#estop { background: #8a2525; border-color: #d04b4b; font-weight: 700; }
QTextEdit { background: #14181c; border: 1px solid #3a424b; color: #dce6ee; }
QLabel#panelTitle { font-size: 16px; font-weight: 700; }
QLabel#visualPlaceholder { background: #151a1f; border: 1px dashed #4b5660; color: #90a0ad; }
QLabel#cameraFrame { background: #0f1317; border: 1px solid #3a424b; color: #90a0ad; }
QLabel#summaryTile { background: #151a1f; border: 1px solid #303943; border-radius: 5px; padding: 8px; }
QLabel#hostAlert { background: #5b1f1f; border: 1px solid #d04b4b; border-radius: 5px; padding: 8px; font-weight: 700; }
QLabel#connectionStatus { background: #151a1f; border-radius: 5px; padding: 8px; font-weight: 700; }
QLabel#actionState { background: #151a1f; padding: 8px; border-radius: 4px; font-family: monospace; }
QLabel#diagnosticStatus { background: #151a1f; border: 1px solid #303943; border-radius: 5px; padding: 8px; font-family: monospace; }
"""


class MainWindow(QMainWindow):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.ui = UiManager(context)
        self.presenter = StatusPresenter(self.ui)
        self.teleop_thread: QThread | None = None
        self.teleop_worker: TeleopWorker | None = None
        self.command_jobs: list[tuple[QThread, CommandWorker]] = []
        self.record_thread: QThread | None = None
        self.record_worker: CommandWorker | None = None
        self.record_control_file = self.context.config.ops_dir / "record_control.txt"
        self.record_motion_file = self.context.config.ops_dir / "record_motion.json"
        self.record_phase_marker_file = self.context.config.ops_dir / "record_phase_markers.jsonl"
        self.record_phase_marker_output: Path | None = None
        self.record_preview_dir = self.context.config.ops_dir / "record_preview"
        self.record_preview_thread: QThread | None = None
        self.record_preview_worker: RecordPreviewWorker | None = None
        self._record_reset_mode = False
        self._record_stop_requested = False
        self._record_force_stop_timer = QTimer(self)
        self._record_force_stop_timer.setSingleShot(True)
        self._record_force_stop_timer.timeout.connect(self._force_stop_record_after_timeout)
        self.eval_thread: QThread | None = None
        self.eval_worker: CommandWorker | None = None
        self.camera_thread: QThread | None = None
        self.camera_worker: CameraWorker | None = None
        self.voice_thread: QThread | None = None
        self.voice_worker: VoiceWorker | None = None
        self.last_camera_frame = None
        self.last_camera_frames = {}
        self._start_teleop_after_camera = False
        self._start_record_after_camera = False
        self._held_base = (0.0, 0.0, 0.0)
        self._held_lift = 0
        self._voice_motion_active = False
        self._pressed_keyboard_keys: set[str] = set()
        self._motion_timer = QTimer(self)
        self._motion_timer.setInterval(100)
        self._motion_timer.timeout.connect(self.refresh_motion_command)

        self.setWindowTitle("AlohaMini 总控界面")
        self.resize(1280, 780)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCentralWidget(self.ui.build())
        self.setStyleSheet(STYLE)
        self.ui.camera_panel.set_enabled_cameras(self._configured_cameras())

        self._connect_signals()
        QApplication.instance().installEventFilter(self)
        self.ui.log_panel.append("INFO", "AlohaMini 总控界面已就绪。")

    def _connect_signals(self) -> None:
        c = self.ui.connection_tab
        c.start_host.clicked.connect(lambda: self.run_command("start_pi_host", self.context.scripts.script("start_pi_host.sh")))
        c.stop_host.clicked.connect(lambda: self.run_command("stop_pi_host", self.context.scripts.script("stop_pi_host.sh")))
        c.status_check.clicked.connect(lambda: self.run_command("status", self.context.scripts.script("status.sh")))
        c.tail_log.clicked.connect(lambda: self.run_command("tail_host_log", self.context.scripts.ssh_tail_host_log()))
        c.apply_pi.clicked.connect(self.apply_pi_target)
        c.save_pi.clicked.connect(self.save_pi_target)
        c.start_teleop.clicked.connect(self.start_gui_teleop)
        c.stop_teleop.clicked.connect(self.stop_gui_teleop)

        t = self.ui.teleop_tab
        t.forward.pressed.connect(lambda: self.hold_keyboard_key("w"))
        t.forward.released.connect(self.stop_motion)
        t.back.pressed.connect(lambda: self.hold_keyboard_key("s"))
        t.back.released.connect(self.stop_motion)
        t.left.pressed.connect(lambda: self.hold_keyboard_key("z"))
        t.left.released.connect(self.stop_motion)
        t.right.pressed.connect(lambda: self.hold_keyboard_key("x"))
        t.right.released.connect(self.stop_motion)
        t.rot_left.pressed.connect(lambda: self.hold_keyboard_key("a"))
        t.rot_left.released.connect(self.stop_motion)
        t.rot_right.pressed.connect(lambda: self.hold_keyboard_key("d"))
        t.rot_right.released.connect(self.stop_motion)
        t.lift_up.pressed.connect(lambda: self.hold_keyboard_key("u"))
        t.lift_up.released.connect(self.stop_motion)
        t.lift_down.pressed.connect(lambda: self.hold_keyboard_key("j"))
        t.lift_down.released.connect(self.stop_motion)
        t.stop.clicked.connect(self.stop_motion)
        t.estop.clicked.connect(self.emergency_stop)
        t.voice_control.toggled.connect(self.set_voice_control_enabled)

        ds = self.ui.dataset_tab
        ds.start_record.clicked.connect(self.start_record)
        ds.finish_episode.clicked.connect(self.finish_record_episode)
        ds.rerecord_episode.clicked.connect(self.rerecord_record_episode)
        ds.restart_episode.clicked.connect(self.restart_record_episode)
        ds.stop_record.clicked.connect(self.stop_record)
        ds.phase_mark_grasp.clicked.connect(lambda: self.mark_record_phase("grasp"))
        ds.phase_mark_scan.clicked.connect(lambda: self.mark_record_phase("scan"))
        ds.phase_mark_place.clicked.connect(lambda: self.mark_record_phase("place"))

        deploy = self.ui.deploy_tab
        deploy.check_model.clicked.connect(self.check_eval_model)
        deploy.start_eval.clicked.connect(self.start_eval)
        deploy.stop_eval.clicked.connect(self.stop_eval)

        cal = self.ui.calibration_tab
        cal.calibrate.clicked.connect(lambda: self.open_terminal("calibrate_leaders.sh"))
        cal.use_existing.clicked.connect(lambda: self.run_command("use_leader_calibration", self.context.scripts.script("use_leader_calibration.sh")))

        diag = self.ui.diagnostics_tab
        diag.status.clicked.connect(lambda: self.run_command("status", self.context.scripts.script("status.sh")))
        diag.local_servos.clicked.connect(lambda: self.run_command("check_local_servos", self.context.scripts.script("check_local_servos.sh")))
        diag.pi_servos.clicked.connect(lambda: self.run_command("check_pi_servos", self.context.scripts.script("check_pi_servos.sh")))
        diag.serial_ports.clicked.connect(lambda: self.run_command("debug_serial_ports", self.context.scripts.script("debug_serial_ports.sh")))
        diag.lift_axis.clicked.connect(lambda: self.run_command("check_lift_axis", self.context.scripts.script("check_lift_axis.sh")))
        diag.host_log.clicked.connect(lambda: self.run_command("tail_host_log", self.context.scripts.ssh_tail_host_log()))
        diag.local_log.clicked.connect(lambda: self.run_command("local_teleop_log", ["bash", "-lc", "tail -120 /tmp/alohamini_teleop.log 2>/dev/null || true"]))

        cam = self.ui.camera_panel
        cam.connect.clicked.connect(self.start_camera)
        cam.disconnect.clicked.connect(self.stop_camera)
        cam.capture.clicked.connect(self.save_camera_frame)
        cam.apply_selection.clicked.connect(self.start_camera)
        cam.save_selection.clicked.connect(self.save_camera_selection)
        cam.source.currentTextChanged.connect(self.set_camera_source)

    def append_log(self, level: str, message: str) -> None:
        self.ui.log_panel.append(level, message)
        self._update_host_alert(message)

    def _busy_with_robot(self) -> bool:
        return any(
            thread is not None
            for thread in [self.teleop_thread, self.record_thread, self.camera_thread, self.eval_thread]
        )

    def _read_pi_target_inputs(self) -> tuple[str, str] | None:
        pi_user = self.ui.connection_tab.pi_user.text().strip()
        pi_host = self.ui.connection_tab.pi_host.text().strip()
        if not pi_user:
            self.ui.log_panel.append("ERROR", "PI_USER 不能为空。")
            return None
        if not pi_host or any(ch.isspace() for ch in pi_host):
            self.ui.log_panel.append("ERROR", "PI_HOST 不能为空，且不能包含空格。")
            return None
        return pi_user, pi_host

    @Slot()
    def apply_pi_target(self) -> None:
        if self._busy_with_robot():
            self.ui.log_panel.append("WARN", "机器人任务运行中，先停止遥操、采集、相机或评估后再切换 Pi 地址。")
            return
        target = self._read_pi_target_inputs()
        if target is None:
            return
        pi_user, pi_host = target
        self.context.set_pi_target(pi_user, pi_host)
        self.ui.link_state.setText(f"网络链路\n目标 {pi_host}")
        self.ui.log_panel.append("INFO", f"已应用连接配置: {pi_user}@{pi_host}")

    @Slot()
    def save_pi_target(self) -> None:
        if self._busy_with_robot():
            self.ui.log_panel.append("WARN", "机器人任务运行中，先停止遥操、采集、相机或评估后再保存 Pi 地址。")
            return
        target = self._read_pi_target_inputs()
        if target is None:
            return
        pi_user, pi_host = target
        try:
            self.context.set_pi_target(pi_user, pi_host)
            self.context.save_pi_target(pi_user, pi_host)
        except Exception as exc:
            self.ui.log_panel.append("ERROR", f"保存 config.env 失败: {exc}")
            return
        self.ui.link_state.setText(f"网络链路\n目标 {pi_host}")
        self.ui.log_panel.append("INFO", f"已保存连接配置到 config.env: {pi_user}@{pi_host}")

    def _update_host_alert(self, message: str) -> None:
        lower = message.lower()
        if "overcurrent" in lower:
            self.show_host_alert("Host 保护停机：检测到底盘/舵机过流。请检查轮子是否卡住、机械阻力和供电。")
        elif "input voltage error" in lower:
            self.show_host_alert("Host 保护停机：检测到输入电压错误。请检查电池、电源线和负载。")
        elif "shutting down alohamini host" in lower:
            if not self.ui.host_alert.isVisible():
                self.show_host_alert("Host 已退出。请查看 Host 日志确认是否为过流、电压或连接异常。")

    def show_host_alert(self, text: str) -> None:
        self.ui.host_alert.setText(text)
        self.ui.host_alert.show()
        self.ui.host_state.setText("树莓派 Host\n告警")

    def clear_host_alert(self) -> None:
        self.ui.host_alert.hide()
        self.ui.host_alert.clear()

    def run_command(self, label: str, command: list[str]) -> None:
        if label in {"start_pi_host", "status"}:
            self.clear_host_alert()
        self.append_log("INFO", f"启动任务: {label}")
        thread = QThread(self)
        worker = CommandWorker(command, str(self.context.config.ops_dir), self.context.config.env, label)
        worker.moveToThread(thread)
        worker.log.connect(self.append_log)
        worker.result.connect(self.handle_command_result)
        worker.finished.connect(lambda code: self.append_log("INFO" if code == 0 else "ERROR", f"{label} exited with {code}"))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(lambda _code, t=thread, w=worker: self._command_finished(t, w))
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)
        self.command_jobs.append((thread, worker))
        thread.start()

    def _command_finished(self, thread: QThread, worker: CommandWorker) -> None:
        self.command_jobs = [(t, w) for t, w in self.command_jobs if t is not thread and w is not worker]

    def _task_camera_enabled(self) -> bool:
        return self.context.config.env.get("ALOHAMINI_TASK_CAMERA", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @Slot(str, int, str)
    def handle_command_result(self, label: str, code: int, output: str) -> None:
        if label in {"start_pi_host", "stop_pi_host", "status"}:
            self._update_summary_from_command(label, code, output)
        if label == "debug_serial_ports":
            self.ui.diagnostic_status.setText(self._serial_debug_summary(code, output))
        elif label in {"check_local_servos", "check_pi_servos", "check_lift_axis", "status"}:
            self.ui.diagnostic_status.setText(self._general_diagnostic_summary(label, code, output))

    def _update_summary_from_command(self, label: str, code: int, output: str) -> None:
        if label == "start_pi_host":
            if code == 0:
                self.ui.host_state.setText("树莓派 Host\n运行中")
                self.ui.link_state.setText("网络链路\n正常")
            else:
                self.ui.host_state.setText("树莓派 Host\n启动失败")
                self.ui.link_state.setText("网络链路\n检查 SSH/日志")
            return
        if label == "stop_pi_host":
            self.ui.host_state.setText("树莓派 Host\n已停止" if code == 0 else "树莓派 Host\n停止失败")
            return
        if label == "status":
            host_ok = "lekiwi_host" in output
            ssh_ok = code == 0
            self.ui.host_state.setText("树莓派 Host\n运行中" if host_ok else "树莓派 Host\n未运行")
            self.ui.link_state.setText("网络链路\n正常" if ssh_ok else "网络链路\n异常")

    def _general_diagnostic_summary(self, label: str, code: int, output: str) -> str:
        if label == "status":
            local_ok = "/dev/am_arm_leader_left" in output and "/dev/am_arm_leader_right" in output
            pi_ok = "/dev/am_arm_follower_left" in output and "/dev/am_arm_follower_right" in output
            host_ok = "lekiwi_host" in output
            lines = ["诊断摘要", "运行状态检查:"]
            lines.append(f"  本机 Leader 串口: {'OK' if local_ok else '未完整识别'}")
            lines.append(f"  树莓派 Follower 串口: {'OK' if pi_ok else '未完整识别'}")
            lines.append(f"  Host 进程: {'运行中' if host_ok else '未运行或未检测到'}")
            next_steps = []
            if not local_ok or not pi_ok:
                next_steps.append("运行“串口定位/舵机扫描”，确认 ttyACM 与 am_arm 映射。")
            if not host_ok:
                next_steps.append("在连接页点击“启动 Host”，再刷新状态。")
            if code != 0:
                next_steps.append("检查 PI_HOST/PI_USER、网络和 SSH。")
            lines.append("下一步:")
            lines.extend(f"  {step}" for step in (next_steps or ["状态基本正常，可以打开相机/遥操/采集。"]))
            return "\n".join(lines)
        if label == "check_lift_axis":
            ok = code == 0 and "FAILED" not in output and "ERROR" not in output
            return "\n".join(
                [
                    "诊断摘要",
                    f"升降轴检查: {'通过' if ok else '需要处理'}",
                    "下一步:",
                    "  通过后再做遥操/采集；失败时先看左侧日志中的端口、ID、方向和反馈。",
                ]
            )
        return f"诊断摘要\n{label}: {'完成' if code == 0 else '失败'}\n下一步:\n  完整输出见左侧日志。"

    def _serial_debug_summary(self, code: int, output: str) -> str:
        online = re.findall(r"Online IDs:\s*\[([^\]]*)\]\s*on\s*(\S+)", output)
        issues = []
        for prefix in ("MISSING", "PERMISSION", "FAILED", "TIMEOUT", "ERROR"):
            for line in output.splitlines():
                if line.startswith(f"{prefix}:"):
                    issues.append(line)
        lines = ["诊断摘要", f"机械臂串口调试: {'完成' if code == 0 else '有错误'}"]
        if online:
            lines.append("扫描到舵机:")
            for ids, port in online[:10]:
                lines.append(f"  {port}: IDs [{ids.strip()}]")
        else:
            lines.append("未扫描到舵机 ID，检查供电、USB、权限或端口占用。")
        if issues:
            lines.append("注意:")
            for issue in issues[:8]:
                lines.append(f"  {issue}")
        lines.append("下一步:")
        if not online:
            lines.append("  确认机械臂供电、USB 连接、dialout 权限；Host 运行时可能占用 Pi 串口。")
        else:
            lines.append("  按 ID 分布确认 am_arm 链接是否指向正确 ttyACM；错误时修 udev 或重新插线。")
        lines.append("完整日志见左侧日志面板。")
        return "\n".join(lines)

    def open_terminal(self, script_name: str) -> None:
        try:
            subprocess.Popen(self.context.scripts.open_terminal_command(script_name), env=self.context.config.env)
            self.ui.log_panel.append("INFO", f"已打开终端: {script_name}")
        except Exception as exc:
            self.ui.log_panel.append("ERROR", f"打开终端失败: {exc}")

    def _validate_resume_dataset(self, dataset_path: Path) -> str | None:
        required = [
            dataset_path / "meta" / "info.json",
            dataset_path / "meta" / "stats.json",
            dataset_path / "meta" / "tasks.parquet",
            dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            dataset_path / "data" / "chunk-000" / "file-000.parquet",
        ]
        missing = [str(path.relative_to(dataset_path)) for path in required if not path.exists()]
        if missing:
            return "已有数据集不完整，缺少: " + ", ".join(missing)

        try:
            info = json.loads((dataset_path / "meta" / "info.json").read_text(encoding="utf-8"))
        except Exception as exc:
            return f"无法读取 meta/info.json: {exc}"

        if int(info.get("total_episodes") or 0) <= 0 or int(info.get("total_frames") or 0) <= 0:
            return (
                "已有数据集尚未完成 finalize，info.json 中 total_episodes/total_frames 仍为 0。"
                "通常是上次采集被强制停止或未完成“复位完成，继续采集”。"
            )

        try:
            import pandas as pd

            pd.read_parquet(dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
            pd.read_parquet(dataset_path / "data" / "chunk-000" / "file-000.parquet")
        except Exception as exc:
            return f"已有数据集 parquet 损坏，不能继续写入: {exc}"

        return None

    @Slot()
    def start_record(self) -> None:
        if self.record_thread is not None:
            self.ui.log_panel.append("WARN", "数据采集已在运行。")
            return
        if self.teleop_thread is not None:
            self.ui.log_panel.append("WARN", "GUI 遥操正在运行。请先断开 GUI 遥操，再启动数据采集。")
            return
        if self.camera_thread is not None:
            self.ui.log_panel.append("INFO", "机器人相机预览正在运行；将先停止预览，再启动数据采集。")
            self._start_record_after_camera = True
            self.stop_camera()
            return
        dataset_id = self.ui.dataset_tab.dataset.text().strip()
        if not dataset_id:
            self.ui.log_panel.append("ERROR", "数据集 repo_id 不能为空。")
            return
        if Path(dataset_id).is_absolute():
            self.ui.log_panel.append("ERROR", "数据集 repo_id 请填写相对名称，例如 local/alohamini_pick_lift_move_test_02。")
            return
        dataset_path = self.context.config.dataset_home / dataset_id
        if dataset_path.exists() and not self.ui.dataset_tab.resume.isChecked():
            self.ui.log_panel.append(
                "ERROR",
                "数据集目录已存在。首次新建请换一个 repo_id；如果要追加到已有数据集，请勾选“继续写入已有数据集”。"
                f" 当前路径: {dataset_path}",
            )
            return
        if self.ui.dataset_tab.resume.isChecked() and not dataset_path.exists():
            self.ui.log_panel.append(
                "ERROR",
                f"勾选了继续写入，但数据集目录不存在: {dataset_path}",
            )
            return
        if self.ui.dataset_tab.resume.isChecked():
            invalid_reason = self._validate_resume_dataset(dataset_path)
            if invalid_reason:
                self.ui.log_panel.append(
                    "ERROR",
                    f"不能继续写入这个数据集: {invalid_reason} 当前路径: {dataset_path}",
                )
                return
        try:
            self.record_control_file.write_text("", encoding="utf-8")
            self.record_motion_file.write_text(
                json.dumps({"keys": [], "stamp": time.time_ns()}),
                encoding="utf-8",
            )
            self.record_phase_marker_file.write_text("", encoding="utf-8")
            self.record_phase_marker_output = None
            phase_specs = self.ui.dataset_tab.phase_marker_specs()
            if phase_specs:
                marker_dir = self.context.config.dataset_home / "_phase_markers"
                marker_dir.mkdir(parents=True, exist_ok=True)
                safe_dataset_id = re.sub(r"[^A-Za-z0-9_.-]+", "__", dataset_id).strip("_")
                session_stamp = time.strftime("%Y%m%d_%H%M%S")
                self.record_phase_marker_output = marker_dir / f"{safe_dataset_id}_{session_stamp}.jsonl"
            self.record_preview_dir.mkdir(parents=True, exist_ok=True)
            for path in self.record_preview_dir.glob("*.jpg"):
                path.unlink()
        except Exception as exc:
            self.ui.log_panel.append("ERROR", f"无法创建采集控制文件: {exc}")
            return
        args = self.ui.dataset_tab.build_args()
        args.extend(["--control_file", str(self.record_control_file)])
        args.extend(["--motion_file", str(self.record_motion_file)])
        args.extend(["--preview_dir", str(self.record_preview_dir)])
        args.extend(["--preview_fps", self.context.config.env.get("ALOHAMINI_RECORD_PREVIEW_FPS", "8")])
        if self.record_phase_marker_output is not None:
            args.extend(["--phase_marker_file", str(self.record_phase_marker_file)])
            args.extend(["--phase_marker_output", str(self.record_phase_marker_output)])
        command = self.context.scripts.script("start_record.sh") + args
        self.record_thread = QThread(self)
        self.record_worker = CommandWorker(
            command,
            str(self.context.config.ops_dir),
            self.context.config.env,
            "record_dataset",
        )
        self.record_worker.moveToThread(self.record_thread)
        self.record_worker.log.connect(self.append_log)
        self.record_worker.finished.connect(self._record_finished)
        self.record_worker.finished.connect(self.record_thread.quit)
        self.record_worker.finished.connect(self.record_worker.deleteLater)
        self.record_thread.finished.connect(self.record_thread.deleteLater)
        self.record_thread.started.connect(self.record_worker.run)
        self.ui.dataset_tab.start_record.setEnabled(False)
        self.ui.dataset_tab.finish_episode.setEnabled(True)
        self.ui.dataset_tab.rerecord_episode.setEnabled(True)
        self.ui.dataset_tab.restart_episode.setEnabled(False)
        self.ui.dataset_tab.stop_record.setEnabled(True)
        self.ui.dataset_tab.stop_record.setText("停止数据采集")
        self._set_phase_marker_buttons_enabled(self.record_phase_marker_output is not None)
        self.ui.data_state.setText("数据采集\n运行中")
        self._record_reset_mode = False
        self._record_stop_requested = False
        self._pressed_keyboard_keys.clear()
        self.ui.log_panel.append("INFO", "数据采集已在 GUI 后台启动。")
        if self.record_phase_marker_output is not None:
            specs_text = ", ".join(f"{spec['key']}={spec['label']}" for spec in self.ui.dataset_tab.phase_marker_specs())
            self.ui.log_panel.append("INFO", f"旁路阶段标注已启用: {specs_text}")
            self.ui.log_panel.append("INFO", f"阶段标注 sidecar: {self.record_phase_marker_output}")
        self.start_record_preview()
        self.record_thread.start()

    @Slot()
    def stop_record(self) -> None:
        if self.record_worker is None:
            self.ui.log_panel.append("WARN", "当前没有正在运行的数据采集。")
            return
        if self._record_stop_requested:
            self.ui.log_panel.append("WARN", "正在强制终止数据采集进程；未完成保存的数据可能损坏。")
            self._record_force_stop_timer.stop()
            self.record_worker.cancel()
            return

        self.ui.log_panel.append("WARN", "已请求停止数据采集；正在等待程序安全保存并退出，请不要关闭窗口。")
        self._record_reset_mode = False
        self._record_stop_requested = True
        self._write_record_motion_keys(set())
        self._write_record_control("stop")
        self.ui.dataset_tab.finish_episode.setEnabled(False)
        self.ui.dataset_tab.rerecord_episode.setEnabled(False)
        self.ui.dataset_tab.restart_episode.setEnabled(False)
        self.ui.dataset_tab.stop_record.setText("强制停止采集")
        self._set_phase_marker_buttons_enabled(False)
        self.ui.data_state.setText("数据采集\n正在安全停止")

        timeout_raw = self.context.config.env.get("ALOHAMINI_RECORD_STOP_TIMEOUT_S", "300")
        try:
            timeout_s = int(timeout_raw)
        except ValueError:
            timeout_s = 300
            self.ui.log_panel.append(
                "WARN",
                f"ALOHAMINI_RECORD_STOP_TIMEOUT_S={timeout_raw!r} 不是有效秒数，使用默认 300 秒。",
            )
        if timeout_s > 0:
            self._record_force_stop_timer.start(timeout_s * 1000)
            self.ui.log_panel.append(
                "INFO",
                f"如果 {timeout_s} 秒后仍未退出，GUI 会执行强制终止。再次点击“强制停止采集”也会立即终止。",
            )

    @Slot()
    def _force_stop_record_after_timeout(self) -> None:
        if self.record_worker is None or not self._record_stop_requested:
            return
        self.ui.log_panel.append("ERROR", "数据采集安全停止超时，正在强制终止；请检查本次数据集是否完整。")
        self.record_worker.cancel()

    @Slot()
    def finish_record_episode(self) -> None:
        if self.record_worker is None:
            self.ui.log_panel.append("WARN", "当前没有正在运行的数据采集。")
            return
        self._write_record_control("finish_wait")
        self._record_reset_mode = True
        self._pressed_keyboard_keys.clear()
        self._write_record_motion_keys(set())
        self.ui.dataset_tab.finish_episode.setEnabled(False)
        self.ui.dataset_tab.rerecord_episode.setEnabled(False)
        self.ui.dataset_tab.restart_episode.setEnabled(True)
        self._set_phase_marker_buttons_enabled(False)
        self.ui.data_state.setText("数据采集\n等待复位")
        self.ui.log_panel.append(
            "INFO",
            "已请求完成当前段并保存；保存完成后进入不记录数据的遥操作复位阶段。复位完成后点击“复位完成，继续采集”。",
        )

    @Slot()
    def rerecord_record_episode(self) -> None:
        if self.record_worker is None:
            self.ui.log_panel.append("WARN", "当前没有正在运行的数据采集。")
            return
        self._write_record_control("rerecord_wait")
        self._record_reset_mode = True
        self._pressed_keyboard_keys.clear()
        self._write_record_motion_keys(set())
        self.ui.dataset_tab.finish_episode.setEnabled(False)
        self.ui.dataset_tab.rerecord_episode.setEnabled(False)
        self.ui.dataset_tab.restart_episode.setEnabled(True)
        self._set_phase_marker_buttons_enabled(False)
        self.ui.data_state.setText("数据采集\n等待复位")
        self.ui.log_panel.append(
            "WARN",
            "已请求废弃当前段；当前段不会保存。等待复位期间仍可遥操作，请把机器人和物体复位到原处，然后点击“复位完成，继续采集”。",
        )

    @Slot()
    def restart_record_episode(self) -> None:
        if self.record_worker is None:
            self.ui.log_panel.append("WARN", "当前没有正在运行的数据采集。")
            return
        self._write_record_control("restart")
        self._record_reset_mode = False
        self._pressed_keyboard_keys.clear()
        self._write_record_motion_keys(set())
        self.ui.dataset_tab.finish_episode.setEnabled(True)
        self.ui.dataset_tab.rerecord_episode.setEnabled(True)
        self.ui.dataset_tab.restart_episode.setEnabled(False)
        self._set_phase_marker_buttons_enabled(self.record_phase_marker_output is not None)
        self.ui.data_state.setText("数据采集\n运行中")
        self.ui.log_panel.append("INFO", "已确认复位完成；record_bi.py 将继续采集。")

    def _write_record_control(self, command: str) -> None:
        try:
            self.record_control_file.write_text(f"{command} {time.time_ns()}", encoding="utf-8")
        except Exception as exc:
            self.ui.log_panel.append("ERROR", f"写入采集控制命令失败: {exc}")

    def _write_record_motion_keys(self, keys: set[str]) -> None:
        try:
            self.record_motion_file.write_text(
                json.dumps({"keys": sorted(keys), "stamp": time.time_ns()}),
                encoding="utf-8",
            )
        except Exception as exc:
            self.ui.log_panel.append("ERROR", f"写入复位遥操作命令失败: {exc}")

    def _set_phase_marker_buttons_enabled(self, enabled: bool) -> None:
        for button in (
            self.ui.dataset_tab.phase_mark_grasp,
            self.ui.dataset_tab.phase_mark_scan,
            self.ui.dataset_tab.phase_mark_place,
        ):
            button.setEnabled(enabled)

    def _phase_marker_spec_by_name(self, name: str) -> dict[str, str] | None:
        for spec in self.ui.dataset_tab.phase_marker_specs():
            if spec["name"] == name:
                return spec
        return None

    def _phase_marker_spec_by_key(self, key: str) -> dict[str, str] | None:
        for spec in self.ui.dataset_tab.phase_marker_specs():
            if spec["key"] == key:
                return spec
        return None

    def _is_text_input_focused(self) -> bool:
        widget = QApplication.focusWidget()
        return isinstance(widget, (QLineEdit, QTextEdit))

    def _phase_key_from_event(self, event) -> str | None:
        if event.isAutoRepeat():
            return None
        text = event.text()
        if not text:
            return None
        return text.strip().lower()

    @Slot(str)
    def mark_record_phase(self, name: str) -> None:
        spec = self._phase_marker_spec_by_name(name)
        if spec is None:
            self.ui.log_panel.append("WARN", f"阶段标注 {name} 未启用或配置不完整。")
            return
        self._write_record_phase_marker(spec)

    def _write_record_phase_marker(self, spec: dict[str, str]) -> None:
        if self.record_worker is None:
            self.ui.log_panel.append("WARN", "当前没有正在运行的数据采集，阶段标注未写入。")
            return
        if self._record_reset_mode:
            self.ui.log_panel.append("WARN", "当前处于复位等待阶段，阶段标注未写入。")
            return
        if self._record_stop_requested:
            self.ui.log_panel.append("WARN", "数据采集正在停止，阶段标注未写入。")
            return
        if self.record_phase_marker_output is None:
            self.ui.log_panel.append("WARN", "旁路阶段标注未启用。")
            return
        payload = {
            "type": "phase_marker",
            "name": spec["name"],
            "label": spec["label"],
            "key": spec["key"],
            "dataset": self.ui.dataset_tab.dataset.text().strip(),
            "wall_time_ns": time.time_ns(),
        }
        try:
            with self.record_phase_marker_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            self.ui.log_panel.append("ERROR", f"写入阶段标注失败: {exc}")
            return
        self.ui.log_panel.append("INFO", f"阶段标注: {spec['label']} -> {self.record_phase_marker_output}")

    @Slot(int)
    def _record_finished(self, code: int) -> None:
        level = "INFO" if code == 0 else "ERROR"
        self.ui.log_panel.append(level, f"record_dataset exited with {code}")
        self.ui.data_state.setText("数据采集\n待机" if code == 0 else "数据采集\n已退出")
        self.ui.dataset_tab.start_record.setEnabled(True)
        self.ui.dataset_tab.finish_episode.setEnabled(False)
        self.ui.dataset_tab.rerecord_episode.setEnabled(False)
        self.ui.dataset_tab.restart_episode.setEnabled(False)
        self.ui.dataset_tab.stop_record.setEnabled(False)
        self.ui.dataset_tab.stop_record.setText("停止数据采集")
        self._set_phase_marker_buttons_enabled(False)
        self._record_force_stop_timer.stop()
        self._record_reset_mode = False
        self._record_stop_requested = False
        self._pressed_keyboard_keys.clear()
        self._write_record_motion_keys(set())
        if self.record_phase_marker_output is not None:
            self.ui.log_panel.append("INFO", f"阶段标注 sidecar 已结束: {self.record_phase_marker_output}")
        self.record_phase_marker_output = None
        self.record_worker = None
        self.record_thread = None
        self.stop_record_preview()

    def _validate_deploy_fields(self) -> list[str] | None:
        tab = self.ui.deploy_tab
        model_path = Path(tab.model_path.text().strip())
        dataset_id = tab.dataset.text().strip()
        if not model_path.exists():
            self.ui.log_panel.append("ERROR", f"模型路径不存在: {model_path}")
            return None
        if not (model_path / "config.json").exists():
            self.ui.log_panel.append("ERROR", f"模型目录缺少 config.json: {model_path}")
            return None
        if not dataset_id:
            self.ui.log_panel.append("ERROR", "评估数据集 repo_id 不能为空。")
            return None
        if Path(dataset_id).is_absolute():
            self.ui.log_panel.append("ERROR", "评估数据集 repo_id 请填写相对名称，例如 local/eval_act_test_01。")
            return None
        for label, value in [
            ("评估段数", tab.num_episodes.text().strip()),
            ("FPS", tab.fps.text().strip()),
            ("每段时长秒", tab.episode_time.text().strip()),
        ]:
            try:
                if int(value) <= 0:
                    raise ValueError
            except ValueError:
                self.ui.log_panel.append("ERROR", f"{label} 必须是正整数。")
                return None
        return tab.build_args(self.context.config.pi_host, self.context.config.robot_model)

    @Slot()
    def check_eval_model(self) -> None:
        model_path = Path(self.ui.deploy_tab.model_path.text().strip())
        if not model_path.exists():
            self.ui.log_panel.append("ERROR", f"模型路径不存在: {model_path}")
            return
        command = self.context.scripts.script("start_eval.sh") + ["--check_model", str(model_path)]
        self.run_command("check_eval_model", command)

    @Slot()
    def start_eval(self) -> None:
        if self.eval_thread is not None:
            self.ui.log_panel.append("WARN", "真机评估已在运行。")
            return
        if self.record_thread is not None or self.teleop_thread is not None or self.camera_thread is not None:
            self.ui.log_panel.append("WARN", "请先停止数据采集、GUI 遥操或相机预览，再启动真机评估。")
            return
        args = self._validate_deploy_fields()
        if args is None:
            return
        command = self.context.scripts.script("start_eval.sh") + args
        self.eval_thread = QThread(self)
        self.eval_worker = CommandWorker(
            command,
            str(self.context.config.ops_dir),
            self.context.config.env,
            "evaluate_policy",
        )
        self.eval_worker.moveToThread(self.eval_thread)
        self.eval_worker.log.connect(self.append_log)
        self.eval_worker.finished.connect(self._eval_finished)
        self.eval_worker.finished.connect(self.eval_thread.quit)
        self.eval_worker.finished.connect(self.eval_worker.deleteLater)
        self.eval_thread.finished.connect(self.eval_thread.deleteLater)
        self.eval_thread.started.connect(self.eval_worker.run)
        self.ui.deploy_tab.start_eval.setEnabled(False)
        self.ui.deploy_tab.stop_eval.setEnabled(True)
        self.ui.robot_state.setText("机器人\n策略评估")
        self.ui.log_panel.append("WARN", f"将使用当前 Pi 地址做真机评估: {self.context.config.pi_host}")
        self.eval_thread.start()

    @Slot()
    def stop_eval(self) -> None:
        if self.eval_worker is None:
            self.ui.log_panel.append("WARN", "当前没有正在运行的真机评估。")
            return
        self.ui.log_panel.append("WARN", "正在停止真机评估进程...")
        self.eval_worker.cancel()

    @Slot(int)
    def _eval_finished(self, code: int) -> None:
        level = "INFO" if code == 0 else "ERROR"
        self.ui.log_panel.append(level, f"evaluate_policy exited with {code}")
        self.ui.deploy_tab.start_eval.setEnabled(True)
        self.ui.deploy_tab.stop_eval.setEnabled(False)
        self.ui.robot_state.setText("机器人\n未连接")
        self.eval_worker = None
        self.eval_thread = None

    @Slot()
    def start_gui_teleop(self) -> None:
        if self.record_thread is not None:
            self.ui.log_panel.append("WARN", "数据采集正在运行。请先停止或完成数据采集，再启动 GUI 遥操。")
            return
        if self.teleop_thread is not None:
            self.ui.log_panel.append("WARN", "GUI 遥操已在运行。")
            return
        if self.camera_thread is not None:
            self.ui.log_panel.append("WARN", "机器人相机会占用 observation，正在先停止相机再启动 GUI 遥操。")
            self._start_teleop_after_camera = True
            self.stop_camera()
            return
        cfg = {
            "pi_host": self.context.config.pi_host,
            "robot_model": self.context.config.robot_model,
            "leader_id": self.context.config.leader_id,
            "arm_profile": self.context.config.arm_profile,
        }
        self.teleop_thread = QThread(self)
        camera_name = self.ui.camera_panel.source.currentText().strip() or "auto"
        self.teleop_worker = TeleopWorker(cfg, self.ui.connection_tab.use_leader.isChecked(), camera_name)
        if self._task_camera_enabled():
            self.teleop_worker.set_camera_enabled(True)
        self.teleop_worker.moveToThread(self.teleop_thread)
        self.teleop_worker.log.connect(self.append_log)
        self.teleop_worker.state.connect(self.presenter.update_action)
        self.teleop_worker.state.connect(self.ui.sensor_panel.update_action)
        self.teleop_worker.observation.connect(self.update_observation)
        self.teleop_worker.connected.connect(self.presenter.set_connected)
        self.teleop_worker.connected.connect(lambda connected: self.clear_host_alert() if connected else None)
        self.teleop_worker.sources.connect(self.ui.camera_panel.set_sources)
        self.teleop_worker.frame.connect(self.update_camera_frame)
        self.teleop_worker.frames.connect(self.update_camera_frames)
        self.teleop_worker.finished.connect(self._teleop_finished)
        self.teleop_worker.finished.connect(self.teleop_thread.quit)
        self.teleop_worker.finished.connect(self.teleop_worker.deleteLater)
        self.teleop_thread.finished.connect(self.teleop_thread.deleteLater)
        self.teleop_thread.started.connect(self.teleop_worker.run)
        self.teleop_thread.start()

    @Slot()
    def start_camera(self) -> None:
        if self.record_thread is not None:
            self.ui.log_panel.append("WARN", "数据采集正在运行。采集进程已使用相机，请不要同时打开相机预览。")
            return
        if not self._apply_camera_selection_if_safe(silent=True):
            return
        if self.ui.camera_panel.source.currentText() != "auto":
            self.ui.camera_panel.source.setCurrentText("auto")
        if self.camera_thread is not None:
            self.ui.log_panel.append("WARN", "相机已在运行。")
            return
        if self.teleop_thread is not None:
            self.set_camera_source()
            if self.teleop_worker is not None:
                self.teleop_worker.set_camera_enabled(True)
            self.ui.log_panel.append("INFO", "已打开遥操内置相机预览。")
            return
        camera_name = self.ui.camera_panel.source.currentText().strip() or "auto"
        cfg = {
            "pi_host": self.context.config.pi_host,
            "robot_model": self.context.config.robot_model,
        }
        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(cfg, camera_name)
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_worker.log.connect(self.append_log)
        self.camera_worker.sources.connect(self.ui.camera_panel.set_sources)
        self.camera_worker.observation.connect(self.update_observation)
        self.camera_worker.frame.connect(self.update_camera_frame)
        self.camera_worker.frames.connect(self.update_camera_frames)
        self.camera_worker.finished.connect(self._camera_finished)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_thread.start()

    def _configured_cameras(self) -> list[str]:
        raw = self.context.config.env.get("ALOHAMINI_CAMERAS", "forward,wrist_right")
        names = [name.strip() for name in raw.split(",") if name.strip()]
        return names or ["forward", "wrist_right"]

    @Slot()
    def apply_camera_selection(self) -> None:
        self._apply_camera_selection_if_safe(silent=False)

    @Slot()
    def save_camera_selection(self) -> None:
        if not self._apply_camera_selection_if_safe(silent=False):
            return
        value = self.context.config.env["ALOHAMINI_CAMERAS"]
        try:
            self.context.save_env_values({"ALOHAMINI_CAMERAS": value})
        except Exception as exc:
            self.ui.log_panel.append("ERROR", f"保存相机配置失败: {exc}")
            return
        self.ui.log_panel.append("INFO", f"已保存相机配置到 config.env: {value}")
        self.ui.visual_status.setText(
            f"已保存相机配置: {value}\n下一步: 停止 Host，再启动 Host；Host 重启后 GUI 才能看到新的相机列表。"
        )

    def _apply_camera_selection_if_safe(self, silent: bool = False) -> bool:
        selected = self.ui.camera_panel.selected_cameras()
        if not selected:
            self.ui.log_panel.append("ERROR", "至少选择一个相机。")
            return False
        if self.camera_thread is not None or self.teleop_thread is not None or self.record_thread is not None:
            self.ui.log_panel.append("WARN", "相机/遥操/采集运行中，先停止当前任务后再切换相机列表。")
            return False
        value = ",".join(selected)
        self.context.config.env["ALOHAMINI_CAMERAS"] = value
        os.environ["ALOHAMINI_CAMERAS"] = value
        self.ui.camera_panel.set_sources(selected)
        if not silent:
            self.ui.log_panel.append("INFO", f"已应用相机启用列表: {value}")
            self.ui.visual_status.setText(
                f"相机启用列表: {value}\n如果树莓派 Host 已经运行，需要先停止 Host 再启动 Host，新的相机列表才会生效。"
            )
        return True

    @Slot()
    def stop_camera(self) -> None:
        if self.teleop_thread is not None:
            if self.teleop_worker is not None:
                self.teleop_worker.set_camera_enabled(False)
            self.ui.camera_panel.clear_frames("遥操内置相机预览已停止")
            self.ui.log_panel.append("INFO", "已停止遥操内置相机预览。")
            return
        if self.camera_worker is not None:
            self.ui.log_panel.append("INFO", "正在停止机器人相机...")
            self.camera_worker.stop()

    @Slot()
    def set_camera_source(self, _name: str | None = None) -> None:
        if self.teleop_worker is not None:
            camera_name = self.ui.camera_panel.source.currentText().strip() or "auto"
            self.teleop_worker.set_camera_name(camera_name)

    @Slot(object)
    def update_camera_frame(self, image) -> None:
        self.last_camera_frame = image
        label = self.ui.camera_panel.first_label()
        if label is None:
            return
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)

    @Slot(object)
    def update_camera_frames(self, images: dict) -> None:
        self.last_camera_frames = dict(images)
        if images:
            self.last_camera_frame = next(iter(images.values()))
        panel = self.ui.camera_panel
        names = list(images.keys())
        panel.set_frame_names(names)
        used_labels = []
        for name, label in panel.labels_for_images(names):
            image = images[name]
            pixmap = QPixmap.fromImage(image)
            scaled = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled)
            used_labels.append(label)
        for label in panel.frames.values():
            if label not in used_labels:
                label.clear()
                label.setText("未收到相机帧")

    @Slot(object)
    def update_observation(self, observation: dict) -> None:
        self.ui.map_panel.update_observation(observation)
        self.ui.sensor_panel.update_observation(observation)

    def start_record_preview(self) -> None:
        if self.record_preview_thread is not None:
            return
        fps = int(self.context.config.env.get("ALOHAMINI_RECORD_PREVIEW_FPS", "8"))
        self.record_preview_thread = QThread(self)
        self.record_preview_worker = RecordPreviewWorker(self.record_preview_dir, fps=fps)
        self.record_preview_worker.moveToThread(self.record_preview_thread)
        self.record_preview_worker.log.connect(self.append_log)
        self.record_preview_worker.frames.connect(self.update_camera_frames)
        self.record_preview_worker.finished.connect(self._record_preview_finished)
        self.record_preview_worker.finished.connect(self.record_preview_thread.quit)
        self.record_preview_worker.finished.connect(self.record_preview_worker.deleteLater)
        self.record_preview_thread.finished.connect(self.record_preview_thread.deleteLater)
        self.record_preview_thread.started.connect(self.record_preview_worker.run)
        self.record_preview_thread.start()

    def stop_record_preview(self) -> None:
        if self.record_preview_worker is not None:
            self.record_preview_worker.stop()

    @Slot()
    def _record_preview_finished(self) -> None:
        self.record_preview_worker = None
        self.record_preview_thread = None

    @Slot()
    def _camera_finished(self) -> None:
        self.camera_worker = None
        self.camera_thread = None
        if self._start_teleop_after_camera:
            self._start_teleop_after_camera = False
            QTimer.singleShot(0, self.start_gui_teleop)
        if self._start_record_after_camera:
            self._start_record_after_camera = False
            QTimer.singleShot(0, self.start_record)

    @Slot()
    def save_camera_frame(self) -> None:
        if self.last_camera_frame is None:
            self.ui.log_panel.append("WARN", "当前没有可保存的相机帧。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存当前帧", "alohamini_camera_frame.png", "Images (*.png *.jpg)")
        if path:
            self.last_camera_frame.save(path)
            self.ui.log_panel.append("INFO", f"已保存相机帧: {path}")

    @Slot()
    def stop_gui_teleop(self) -> None:
        if self.teleop_worker is not None:
            self.teleop_worker.stop()

    @Slot()
    def _teleop_finished(self) -> None:
        self.teleop_worker = None
        self.teleop_thread = None
        self.presenter.set_connected(False)

    @Slot(bool)
    def set_voice_control_enabled(self, enabled: bool) -> None:
        if enabled:
            self.start_voice_control()
        else:
            self.stop_voice_control()

    def start_voice_control(self) -> None:
        if self.voice_thread is not None:
            return
        model_name = self.context.config.env.get("ALOHAMINI_VOICE_MODEL", "small")
        device_index = self.context.config.env.get("VOICE_DEVICE_INDEX")
        self.voice_thread = QThread(self)
        self.voice_worker = VoiceWorker(model_name=model_name, device_index=device_index)
        self.voice_worker.moveToThread(self.voice_thread)
        self.voice_worker.log.connect(self.append_log)
        self.voice_worker.heard.connect(lambda text: self.append_log("INFO", f"识别到语音: {text}"))
        self.voice_worker.command.connect(self.handle_voice_command)
        self.voice_worker.finished.connect(self._voice_finished)
        self.voice_worker.finished.connect(self.voice_thread.quit)
        self.voice_worker.finished.connect(self.voice_worker.deleteLater)
        self.voice_thread.finished.connect(self.voice_thread.deleteLater)
        self.voice_thread.started.connect(self.voice_worker.run)
        self.voice_thread.start()
        self.ui.log_panel.append("INFO", "正在启动语音控制...")

    def stop_voice_control(self) -> None:
        if self.voice_worker is not None:
            self.voice_worker.stop()

    @Slot()
    def _voice_finished(self) -> None:
        self.voice_worker = None
        self.voice_thread = None
        if self.ui.teleop_tab.voice_control.isChecked():
            self.ui.teleop_tab.voice_control.blockSignals(True)
            self.ui.teleop_tab.voice_control.setChecked(False)
            self.ui.teleop_tab.voice_control.blockSignals(False)
        self.ui.log_panel.append("INFO", "语音控制已停止。")

    @Slot(object)
    def handle_voice_command(self, command: dict) -> None:
        kind = command.get("kind")
        name = command.get("name")
        text = command.get("text", "")
        self.ui.log_panel.append("INFO", f"语音命令: {name} ({text})")
        if kind == "emergency_stop":
            self.emergency_stop()
            return
        if kind == "record":
            self._handle_voice_record_command(name)
            return
        if kind == "motion":
            self._apply_voice_motion(name)

    def _handle_voice_record_command(self, name: str) -> None:
        if name == "finish_wait":
            self.finish_record_episode()
        elif name == "rerecord_wait":
            self.rerecord_record_episode()
        elif name == "restart":
            self.restart_record_episode()
        elif name == "stop":
            self.stop_record()

    def _apply_voice_motion(self, name: str) -> None:
        action = low_speed_motion_action(name)
        x = float(action.get("x.vel", 0.0))
        y = float(action.get("y.vel", 0.0))
        theta = float(action.get("theta.vel", 0.0))
        lift = int(action.get("lift_axis.vel", 0))
        if self.record_worker is not None:
            if not self._record_reset_mode:
                self.ui.log_panel.append("WARN", "采集记录中只允许在复位等待阶段用语音遥操作。")
                return
            key = MOTION_KEY_BY_COMMAND.get(name)
            if key:
                self._voice_motion_active = True
                self._pressed_keyboard_keys = {key}
                self._write_record_motion_keys(self._pressed_keyboard_keys)
            return
        if self.teleop_worker is None:
            self.ui.log_panel.append("WARN", "GUI 遥操未运行，已忽略语音运动命令。")
            return
        self._voice_motion_active = True
        if lift:
            self.hold_lift(lift)
        else:
            self.hold_base(x, y, theta)

    def set_base(self, x: float, y: float, theta: float) -> None:
        if self.teleop_worker is not None:
            self.teleop_worker.set_estop(False)
            self.teleop_worker.set_base(x, y, theta)

    def set_lift(self, vel: int) -> None:
        if self.teleop_worker is not None:
            self.teleop_worker.set_estop(False)
            self.teleop_worker.set_lift(vel)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            return self._handle_key_press(event)
        if event.type() == QEvent.Type.KeyRelease:
            return self._handle_key_release(event)
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if self._handle_key_press(event):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if self._handle_key_release(event):
            return
        super().keyReleaseEvent(event)

    def _handle_key_press(self, event) -> bool:
        if self.record_worker is not None and not self._is_text_input_focused():
            phase_key = self._phase_key_from_event(event)
            if phase_key is not None:
                phase_spec = self._phase_marker_spec_by_key(phase_key)
                if phase_spec is not None:
                    self._write_record_phase_marker(phase_spec)
                    event.accept()
                    return True

        key = self._teleop_key_from_event(event)
        if key is None:
            return False
        if self.record_worker is not None:
            if self._record_reset_mode and key == " ":
                self._pressed_keyboard_keys.clear()
                self._write_record_motion_keys(set())
            elif self._record_reset_mode and key not in {"escape", "q"}:
                self._pressed_keyboard_keys.add(key)
                self._write_record_motion_keys(self._pressed_keyboard_keys)
            event.accept()
            return True
        if key == " ":
            self.stop_motion()
        elif key == "escape":
            self.emergency_stop()
        elif key == "q":
            self.stop_gui_teleop()
        else:
            if self.teleop_worker is None:
                return False
            self._pressed_keyboard_keys.add(key)
            self.refresh_keyboard_keys()
            if not self._motion_timer.isActive():
                self._motion_timer.start()
        event.accept()
        return True

    def _handle_key_release(self, event) -> bool:
        key = self._teleop_key_from_event(event)
        if key is None:
            return False
        if self.record_worker is not None:
            if self._record_reset_mode and key == " ":
                self._pressed_keyboard_keys.clear()
                self._write_record_motion_keys(set())
            elif self._record_reset_mode and key not in {"escape", "q"}:
                self._pressed_keyboard_keys.discard(key)
                self._write_record_motion_keys(self._pressed_keyboard_keys)
            event.accept()
            return True
        if self.teleop_worker is None and key not in {" ", "escape", "q"}:
            return False
        self._pressed_keyboard_keys.discard(key)
        self.refresh_keyboard_keys()
        if not self._pressed_keyboard_keys and not any(self._held_base) and not self._held_lift:
            self._motion_timer.stop()
        event.accept()
        return True

    def _teleop_key_from_event(self, event) -> str | None:
        if event.isAutoRepeat():
            return None
        key = event.key()
        if key == Qt.Key.Key_Space:
            return " "
        if key == Qt.Key.Key_Escape:
            return "escape"
        text = event.text().lower()
        if text in {"w", "s", "z", "x", "a", "d", "u", "j", "r", "f", "q"}:
            return text
        return None

    def refresh_keyboard_keys(self) -> None:
        if self.teleop_worker is not None:
            self.teleop_worker.set_estop(False)
            self.teleop_worker.set_keyboard_keys(self._pressed_keyboard_keys)

    def hold_base(self, x: float, y: float, theta: float) -> None:
        self._held_base = (x, y, theta)
        self._held_lift = 0
        self.refresh_motion_command()
        if not self._motion_timer.isActive():
            self._motion_timer.start()

    def hold_lift(self, vel: int) -> None:
        self._held_base = (0.0, 0.0, 0.0)
        self._held_lift = vel
        self.refresh_motion_command()
        if not self._motion_timer.isActive():
            self._motion_timer.start()

    def hold_keyboard_key(self, key: str) -> None:
        if self.record_worker is not None and self._record_reset_mode:
            self._pressed_keyboard_keys = {key}
            self._write_record_motion_keys(self._pressed_keyboard_keys)
            return
        self._held_base = (0.0, 0.0, 0.0)
        self._held_lift = 0
        self._pressed_keyboard_keys = {key}
        self.refresh_keyboard_keys()
        if not self._motion_timer.isActive():
            self._motion_timer.start()

    @Slot()
    def refresh_motion_command(self) -> None:
        if self.teleop_worker is None and not (self.record_worker is not None and self._record_reset_mode):
            self._motion_timer.stop()
            return
        if self._pressed_keyboard_keys:
            self.refresh_keyboard_keys()
        x, y, theta = self._held_base
        if self.record_worker is not None and self._record_reset_mode:
            # During record reset, motion commands are normally key-based. Voice
            # low-speed continuous motion is intentionally kept in this GUI layer.
            if self._voice_motion_active:
                self._write_record_motion_keys(set())
            return
        if any((x, y, theta)) or self._voice_motion_active:
            self.set_base(x, y, theta)
        if self._held_lift or self._voice_motion_active:
            self.set_lift(self._held_lift)
        if not self._pressed_keyboard_keys and not any(self._held_base) and not self._held_lift:
            self._motion_timer.stop()

    def stop_motion(self) -> None:
        self._voice_motion_active = False
        self._held_base = (0.0, 0.0, 0.0)
        self._held_lift = 0
        self._pressed_keyboard_keys.clear()
        self._motion_timer.stop()
        if self.record_worker is not None and self._record_reset_mode:
            self._write_record_motion_keys(set())
            return
        if self.teleop_worker is not None:
            self.teleop_worker.set_base(0.0, 0.0, 0.0)
            self.teleop_worker.set_lift(0)
            self.teleop_worker.set_keyboard_keys(set())

    def emergency_stop(self) -> None:
        self.stop_motion()
        if self.teleop_worker is not None:
            self.teleop_worker.set_estop(True)
            self.teleop_worker.stop()
        self.ui.log_panel.append("WARN", "已触发急停。")

    def closeEvent(self, event) -> None:
        if self.record_worker is not None:
            if not self._record_stop_requested:
                self.ui.log_panel.append("WARN", "数据采集仍在运行；已请求安全停止。保存完成退出后再关闭窗口。")
                self.stop_record()
            else:
                self.ui.log_panel.append("WARN", "数据采集正在安全停止；保存完成退出后再关闭窗口。")
            event.ignore()
            return
        self._start_teleop_after_camera = False
        self._start_record_after_camera = False
        if self.teleop_worker is not None:
            self.emergency_stop()
            self.stop_gui_teleop()
        if self.camera_worker is not None:
            self.stop_camera()
        if self.record_preview_worker is not None:
            self.stop_record_preview()
        if self.voice_worker is not None:
            self.stop_voice_control()
            if self.voice_thread is not None:
                self.voice_thread.quit()
        for thread, _worker in list(self.command_jobs):
            _worker.cancel()
            thread.quit()
        for thread, _worker in list(self.command_jobs):
            if not thread.wait(800):
                thread.terminate()
                thread.wait(300)
        if self.camera_thread is not None:
            self.camera_thread.quit()
            if not self.camera_thread.wait(800):
                self.camera_thread.terminate()
                self.camera_thread.wait(300)
        if self.record_preview_thread is not None:
            if not self.record_preview_thread.wait(800):
                self.record_preview_thread.terminate()
                self.record_preview_thread.wait(300)
        if self.teleop_thread is not None:
            self.teleop_thread.quit()
            if not self.teleop_thread.wait(1000):
                self.teleop_thread.terminate()
                self.teleop_thread.wait(300)
        if self.voice_thread is not None:
            if not self.voice_thread.wait(1000):
                self.voice_thread.terminate()
                self.voice_thread.wait(300)
        if self.record_thread is not None:
            if not self.record_thread.wait(1000):
                self.record_thread.terminate()
                self.record_thread.wait(300)
        event.accept()
