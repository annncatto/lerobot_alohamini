import threading
import time

from examples.alohamini.evaluate_bi_remote import LatestObservationSender, build_parser


class BlockingObservationClient:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.observations = []

    def control_loop_observation(
        self,
        task,
        verbose=False,
        raw_observation=None,
        observation_timestamp=None,
    ):
        if not self.observations:
            self.started.set()
            assert self.release.wait(timeout=2)
        self.observations.append((raw_observation, observation_timestamp, task, verbose))


def test_latest_observation_sender_replaces_stale_pending_frame():
    client = BlockingObservationClient()
    sender = LatestObservationSender(client, "task")
    sender.start()

    sender.submit({"frame": 1}, time.time())
    assert client.started.wait(timeout=2)

    # Frame 1 is in flight. Frame 3 must replace pending frame 2 instead of waiting behind it.
    sender.submit({"frame": 2}, time.time())
    sender.submit({"frame": 3}, time.time())
    client.release.set()

    deadline = time.monotonic() + 2
    while sender.snapshot()["sent"] < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    sender.stop()

    assert [item[0]["frame"] for item in client.observations] == [1, 3]
    stats = sender.snapshot()
    assert stats["submitted"] == 3
    assert stats["sent"] == 2
    assert stats["replaced"] == 1
    assert stats["error"] is None


def test_remote_eval_defaults_to_latest_observations_and_supports_direct_hardware():
    args = build_parser().parse_args(
        [
            "--policy.path=/checkpoint",
            "--robot.transport=direct",
            "--observation.jpeg_quality=80",
        ]
    )

    assert args.robot_transport == "direct"
    assert args.observation_send_mode == "latest"
    assert args.image_compression_quality == 80
