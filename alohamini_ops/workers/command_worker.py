import os
import signal
import subprocess

from qt_compat import QObject, Signal, Slot


class CommandWorker(QObject):
    log = Signal(str, str)
    finished = Signal(int)

    def __init__(self, command: list[str], cwd: str, env: dict[str, str], label: str):
        super().__init__()
        self.command = command
        self.cwd = cwd
        self.env = env
        self.label = label
        self.proc: subprocess.Popen | None = None

    @Slot()
    def run(self) -> None:
        self.log.emit("INFO", f"$ {self.label}")
        try:
            self.proc = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log.emit("INFO", line.rstrip())
            code = self.proc.wait()
        except Exception as exc:
            self.log.emit("ERROR", f"{self.label}: {exc}")
            code = 1
        self.finished.emit(code)

    @Slot()
    def cancel(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.log.emit("WARN", f"正在终止任务: {self.label}")
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                self.proc.terminate()
