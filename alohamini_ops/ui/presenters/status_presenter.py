class StatusPresenter:
    def __init__(self, ui):
        self.ui = ui

    def set_connected(self, connected: bool) -> None:
        state = "已连接" if connected else "未连接"
        self.ui.connection_tab.status.setText(state)
        self.ui.robot_state.setText(f"机器人\n{state}")
        self.ui.host_state.setText("树莓派 Host\n运行中" if connected else "树莓派 Host\n未知")
        self.ui.link_state.setText("网络链路\n正常" if connected else "网络链路\n未知")
        self.ui.status_line.setText(f"机器人: {state} | Host: {'运行中' if connected else '未知'} | FPS: 30 | 最近动作: --")

    def update_action(self, action: dict) -> None:
        text = (
            f"x={float(action.get('x.vel', 0.0)): .3f}  "
            f"y={float(action.get('y.vel', 0.0)): .3f}  "
            f"theta={float(action.get('theta.vel', 0.0)): .1f}  "
            f"lift={action.get('lift_axis.vel', 0)}"
        )
        self.ui.action_state.setText(text)
        self.ui.status_line.setText(f"机器人: 已连接 | Host: 运行中 | FPS: 30 | 最近动作: {text}")
