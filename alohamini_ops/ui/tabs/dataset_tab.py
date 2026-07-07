from qt_compat import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DatasetTab(QWidget):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.dataset = QLineEdit("local/alohamini_pick_lift_move_test_01")
        self.num_episodes = QLineEdit("1")
        self.fps = QLineEdit("25")
        self.episode_time = QLineEdit("45")
        self.reset_time = QLineEdit("8")
        self.task_description = QLineEdit("pick up object, lift it, then move")
        self.push_to_hub = QCheckBox("采集后上传到 Hugging Face Hub")
        self.push_to_hub.setChecked(False)
        self.resume = QCheckBox("继续写入已有数据集")
        self.phase_markers_enabled = QCheckBox("启用旁路阶段标注，不写入 LeRobot 主数据")
        self.phase_markers_enabled.setChecked(True)
        self.phase_key_grasp = QLineEdit("1")
        self.phase_label_grasp = QLineEdit("start_of_grasp")
        self.phase_key_scan = QLineEdit("2")
        self.phase_label_scan = QLineEdit("start_of_scan")
        self.phase_key_place = QLineEdit("3")
        self.phase_label_place = QLineEdit("start_of_place")
        self.phase_mark_grasp = QPushButton("标注抓取开始")
        self.phase_mark_scan = QPushButton("标注扫描开始")
        self.phase_mark_place = QPushButton("标注放置开始")
        for button in (self.phase_mark_grasp, self.phase_mark_scan, self.phase_mark_place):
            button.setEnabled(False)
        self.start_record = QPushButton("开始数据采集")
        self.finish_episode = QPushButton("完成当前段并保存")
        self.finish_episode.setEnabled(False)
        self.rerecord_episode = QPushButton("废弃当前段并等待复位")
        self.rerecord_episode.setEnabled(False)
        self.restart_episode = QPushButton("复位完成，继续采集")
        self.restart_episode.setEnabled(False)
        self.stop_record = QPushButton("停止数据采集")
        self.stop_record.setEnabled(False)
        self.dataset_home = QLabel(str(context.config.dataset_home))
        self.dataset_home.setWordWrap(True)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "采集入口会调用 examples/alohamini/record_bi.py，保持与命令行采集逻辑一致。"
            "运行日志会显示在 GUI 日志面板中。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        startup_hint = QLabel(
            "初始流程：先启动树莓派 Host；确认 Leader 校准 ID 与当前配置一致；"
            "第一次用新 repo_id 采 1 段；确认回放/可视化正常后，再勾选继续写入逐段追加。"
        )
        startup_hint.setWordWrap(True)
        layout.addWidget(startup_hint)

        keys_hint = QLabel(
            "采集控制：Leader 主臂控制机械臂；W/S 前后，Z/X 左右平移，A/D 旋转，U/J 升降。"
            "右方向键提前结束当前段，左方向键丢弃并重录当前段，Esc 停止整次采集。"
            "GUI 可保存当前段或废弃当前段；之后都能手动复位，再点继续采集。"
        )
        keys_hint.setWordWrap(True)
        layout.addWidget(keys_hint)

        append_hint = QLabel(
            "默认按单条采集：采集段数 1，FPS 25，每段 45 秒，复位 8 秒。"
            "正式稳定后再把采集段数改大批量采集。"
            "已存在的 repo_id 需要勾选继续写入；重新试采请换一个新 repo_id。"
        )
        append_hint.setWordWrap(True)
        layout.addWidget(append_hint)

        grid = QGridLayout()
        rows = [
            ("数据集 repo_id", self.dataset),
            ("采集段数", self.num_episodes),
            ("FPS", self.fps),
            ("每段时长秒", self.episode_time),
            ("重置时长秒", self.reset_time),
            ("任务描述", self.task_description),
            ("单独保存根目录", self.dataset_home),
        ]
        for row, (label, widget) in enumerate(rows):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        layout.addLayout(grid)

        phase_box = QGroupBox("旁路阶段标注")
        phase_layout = QVBoxLayout(phase_box)
        phase_hint = QLabel(
            "采集中按热键或按钮写入独立 jsonl 标注文件；不会改变图像、动作、状态、task 或 LeRobot metadata。"
        )
        phase_hint.setWordWrap(True)
        phase_layout.addWidget(phase_hint)
        phase_layout.addWidget(self.phase_markers_enabled)
        phase_grid = QGridLayout()
        phase_grid.addWidget(QLabel("热键"), 0, 0)
        phase_grid.addWidget(QLabel("阶段文本"), 0, 1)
        phase_grid.addWidget(QLabel("按钮"), 0, 2)
        phase_rows = [
            (self.phase_key_grasp, self.phase_label_grasp, self.phase_mark_grasp),
            (self.phase_key_scan, self.phase_label_scan, self.phase_mark_scan),
            (self.phase_key_place, self.phase_label_place, self.phase_mark_place),
        ]
        for row, (key_widget, label_widget, button) in enumerate(phase_rows, start=1):
            phase_grid.addWidget(key_widget, row, 0)
            phase_grid.addWidget(label_widget, row, 1)
            phase_grid.addWidget(button, row, 2)
        phase_layout.addLayout(phase_grid)
        layout.addWidget(phase_box)

        layout.addWidget(self.resume)
        layout.addWidget(self.push_to_hub)
        layout.addWidget(self.start_record)
        layout.addWidget(self.finish_episode)
        layout.addWidget(self.rerecord_episode)
        layout.addWidget(self.restart_episode)
        layout.addWidget(self.stop_record)
        layout.addStretch(1)

    def build_args(self) -> list[str]:
        args = [
            "--dataset",
            self.dataset.text().strip(),
            "--num_episodes",
            self.num_episodes.text().strip(),
            "--fps",
            self.fps.text().strip(),
            "--episode_time",
            self.episode_time.text().strip(),
            "--reset_time",
            self.reset_time.text().strip(),
            "--task_description",
            self.task_description.text().strip(),
        ]
        if self.resume.isChecked():
            args.append("--resume")
        args.extend(["--push_to_hub", "true" if self.push_to_hub.isChecked() else "false"])
        return args

    def phase_marker_specs(self) -> list[dict[str, str]]:
        if not self.phase_markers_enabled.isChecked():
            return []
        specs = [
            ("grasp", self.phase_key_grasp.text(), self.phase_label_grasp.text()),
            ("scan", self.phase_key_scan.text(), self.phase_label_scan.text()),
            ("place", self.phase_key_place.text(), self.phase_label_place.text()),
        ]
        result = []
        used_keys = set()
        for name, key, label in specs:
            key = key.strip().lower()
            label = label.strip()
            if not key or not label or key in used_keys:
                continue
            used_keys.add(key)
            result.append({"name": name, "key": key, "label": label})
        return result
