import time
import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.alohamini import AlohaMiniClient, AlohaMiniClientConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME
from lerobot.utils.robot_utils import precise_sleep

parser = argparse.ArgumentParser(description="Replay a LeRobot dataset episode")
parser.add_argument("--dataset.repo_id", "--dataset", dest="dataset_repo_id", type=str, required=True,
                    help="Dataset repo_id, e.g. liyitenga/record_20250914225057")
parser.add_argument("--dataset.root", "--root", dest="dataset_root", type=str, default=None,
                    help="Local dataset root. Defaults to $HF_LEROBOT_HOME/<dataset.repo_id> and never downloads from Hub.")
parser.add_argument("--dataset.episode", "--episode", dest="dataset_episode", type=int, default=0,
                    help="Episode index to replay (default 0)")
parser.add_argument(
    "--robot.remote_ip",
    "--remote_ip",
    dest="remote_ip",
    type=str,
    default="127.0.0.1",
    help="AlohaMini host IP address",
)
parser.add_argument("--robot.id", "--robot_id", dest="robot_id", type=str, default="my_alohamini", help="Robot ID")
parser.add_argument(
    "--robot.robot_model",
    "--robot_model",
    dest="robot_model",
    type=str,
    default="alohamini1",
    choices=["alohamini1", "alohamini2", "alohamini2pro"],
    help="AlohaMini model. Must match the --robot_model used on the Pi host side.",
)



args = parser.parse_args()


robot_config = AlohaMiniClientConfig(remote_ip=args.remote_ip, id=args.robot_id,
                                     robot_model=args.robot_model)
robot = AlohaMiniClient(robot_config)


#dataset = LeRobotDataset("liyitenga/record_20250914225057", episodes=[EPISODE_IDX])
dataset_root = Path(args.dataset_root) if args.dataset_root else HF_LEROBOT_HOME / args.dataset_repo_id
info_path = dataset_root / "meta" / "info.json"
if not info_path.exists():
    raise FileNotFoundError(
        f"Local dataset metadata not found: {info_path}\n"
        "This replay script is configured to use local datasets only. "
        "Pass --root /path/to/dataset or make sure the dataset exists under "
        f"{HF_LEROBOT_HOME}."
    )

dataset = LeRobotDataset(args.dataset_repo_id, root=dataset_root, episodes=[args.dataset_episode])
actions = dataset.hf_dataset.select_columns(ACTION)
#print(f"Dataset loaded with id: {dataset.repo_id}, num_frames: {dataset.num_frames}")

robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

#log_say(f"Replaying episode {args.episode} from {args.dataset}")
print(f"Replaying episode {args.dataset_episode} from {args.dataset_repo_id}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx][ACTION][i]) for i, name in enumerate(dataset.features[ACTION]["names"])
    }

    print(f"replay_bi.action:{action}")
    robot.send_action(action)

    precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()
