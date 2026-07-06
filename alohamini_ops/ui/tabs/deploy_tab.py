from datetime import datetime

from qt_compat import QCheckBox, QGridLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


class DeployTab(QWidget):
    def __init__(self, context):
        super().__init__()
        self.context = context
        default_model = (
            context.config.local_repo
            / "outputs"
            / "train"
            / "act_alohamini_full_fixed"
            / "checkpoints"
            / "100000"
            / "pretrained_model"
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_path = QLineEdit(str(default_model))
        self.dataset = QLineEdit(f"local/eval_act_alohamini_{stamp}")
        self.num_episodes = QLineEdit("1")
        self.fps = QLineEdit("25")
        self.episode_time = QLineEdit("15")
        self.task_description = QLineEdit("pick up object, lift it, then move")
        self.push_to_hub = QCheckBox("评估后上传到 Hugging Face Hub")
        self.push_to_hub.setChecked(False)
        self.check_model = QPushButton("检查模型")
        self.start_eval = QPushButton("开始真机评估")
        self.stop_eval = QPushButton("停止评估")
        self.stop_eval.setEnabled(False)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "真机部署第一版使用短时评估 rollout：加载训练好的 checkpoint，连接当前树莓派 IP，"
            "把策略动作发给机器人，并把评估过程保存成本地 LeRobot 数据集。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        safety = QLabel("建议先 1 段、10-15 秒低风险测试；默认不上传 Hub。")
        safety.setWordWrap(True)
        layout.addWidget(safety)

        grid = QGridLayout()
        rows = [
            ("模型路径", self.model_path),
            ("评估数据集 repo_id", self.dataset),
            ("评估段数", self.num_episodes),
            ("FPS", self.fps),
            ("每段时长秒", self.episode_time),
            ("任务描述", self.task_description),
        ]
        for row, (label, widget) in enumerate(rows):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        layout.addLayout(grid)
        layout.addWidget(self.push_to_hub)
        layout.addWidget(self.check_model)
        layout.addWidget(self.start_eval)
        layout.addWidget(self.stop_eval)
        layout.addStretch(1)

    def build_args(self, remote_ip: str, robot_model: str) -> list[str]:
        return [
            "--hf_model_id",
            self.model_path.text().strip(),
            "--hf_dataset_id",
            self.dataset.text().strip(),
            "--num_episodes",
            self.num_episodes.text().strip(),
            "--fps",
            self.fps.text().strip(),
            "--episode_time",
            self.episode_time.text().strip(),
            "--task_description",
            self.task_description.text().strip(),
            "--remote_ip",
            remote_ip,
            "--robot_model",
            robot_model,
            "--push_to_hub",
            "true" if self.push_to_hub.isChecked() else "false",
        ]
