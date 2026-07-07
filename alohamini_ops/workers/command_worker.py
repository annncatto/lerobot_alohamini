import os
import signal
import subprocess

from qt_compat import QObject, Signal, Slot


class CommandWorker(QObject):
    log = Signal(str, str)
    finished = Signal(int)
    result = Signal(str, int, str)

    def __init__(self, command: list[str], cwd: str, env: dict[str, str], label: str):
        super().__init__()
        self.command = command
        self.cwd = cwd
        self.env = env
        self.label = label
        self.proc: subprocess.Popen | None = None
        self._lines: list[str] = []

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
                text = line.rstrip()
                self._lines.append(text)
                self.log.emit("INFO", text)
            code = self.proc.wait()
        except Exception as exc:
            text = f"{self.label}: {exc}"
            self._lines.append(text)
            self.log.emit("ERROR", text)
            code = 1
        self.result.emit(self.label, code, "\n".join(self._lines))
        self.finished.emit(code)

    @Slot()
    def cancel(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.log.emit("WARN", f"正在终止任务: {self.label}")
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                self.proc.terminate()
