#!/usr/bin/env python3

import argparse
import inspect
import time

import lerobot.robots.alohamini  # noqa: F401 — registers alohamini_client robot type
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini import AlohaMiniClient, AlohaMiniClientConfig
from lerobot.rollout.inference.factory import (
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.action_quantization import snap_planar_velocity
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import auto_select_torch_device
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlohaMini robot with a pretrained policy")
    parser.add_argument("--eval.n_episodes", "--num_episodes", dest="num_episodes", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--eval.episode_time_s", "--episode_time", dest="episode_time", type=int, default=60)
    parser.add_argument(
        "--dataset.single_task",
        "--task_description",
        dest="task_description",
        type=str,
        default="robot task",
    )
    parser.add_argument("--policy.path", "--hf_model_id", dest="policy_path", type=str, required=True)
    parser.add_argument("--dataset.repo_id", "--hf_dataset_id", dest="dataset_repo_id", type=str)
    parser.add_argument(
        "--eval.record_dataset",
        dest="record_dataset",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Record observations and actions as a LeRobotDataset during evaluation.",
    )
    parser.add_argument(
        "--dataset.push_to_hub",
        dest="push_to_hub",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to upload the evaluation dataset to Hugging Face Hub.",
    )
    parser.add_argument("--robot.remote_ip", "--remote_ip", dest="remote_ip", type=str, default="127.0.0.1")
    parser.add_argument("--robot.id", "--robot_id", dest="robot_id", type=str, default="my_alohamini")
    parser.add_argument(
        "--robot.robot_model",
        "--robot_model",
        dest="robot_model",
        type=str,
        default="alohamini1",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
        help="Must match the robot_model on the Pi host side",
    )
    parser.add_argument(
        "--inference.type",
        "--inference_type",
        dest="inference_type",
        type=str,
        default="sync",
        choices=["sync", "rtc"],
        help="sync: one policy call per control tick. rtc: async background inference "
        "(only for policies whose predict_action_chunk supports inference_delay/prev_chunk_left_over, "
        "e.g. pi0/pi05/pi0_fast/smolvla/evo1/molmoact2 — not act/diffusion/vqbet).",
    )
    parser.add_argument(
        "--inference.rtc.execution_horizon",
        dest="rtc_execution_horizon",
        type=int,
        default=10,
        help="RTC only: number of steps re-generated per inference call.",
    )
    parser.add_argument(
        "--inference.rtc.max_guidance_weight",
        dest="rtc_max_guidance_weight",
        type=float,
        default=10.0,
        help="RTC only: max guidance weight for prefix inpainting.",
    )
    parser.add_argument(
        "--inference.rtc.queue_threshold",
        dest="rtc_queue_threshold",
        type=int,
        default=30,
        help="RTC only: trigger a new inference call once the action queue drops to this size.",
    )
    parser.add_argument(
        "--interpolation_multiplier",
        dest="interpolation_multiplier",
        type=int,
        default=1,
        help="Send N interpolated actions per policy action for a smoother, higher-rate control loop "
        "(1 = disabled).",
    )
    parser.add_argument(
        "--action.base_snap_speed",
        dest="base_snap_speed",
        type=float,
        default=0.15,
        help="Snap x.vel/y.vel to {-speed, 0, +speed}; set 0 to disable.",
    )
    parser.add_argument(
        "--action.base_snap_deadband",
        dest="base_snap_deadband",
        type=float,
        default=0.05,
        help="Predicted |x.vel|/|y.vel| at or below this value is sent as zero.",
    )
    args = parser.parse_args()
    if args.base_snap_speed < 0:
        raise ValueError("--action.base_snap_speed cannot be negative")
    if args.base_snap_deadband < 0:
        raise ValueError("--action.base_snap_deadband cannot be negative")
    if args.record_dataset and not args.dataset_repo_id:
        raise ValueError("--dataset.repo_id is required when --eval.record_dataset=true")

    device = str(auto_select_torch_device())

    # === Policy ===
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(args.policy_path, config=policy_cfg)

    # === Inference engine config (needs to be set up before .to(device)/.eval(), mirroring
    # lerobot.rollout.context.build_rollout_context) ===
    if args.inference_type == "rtc":
        predict_chunk_params = inspect.signature(policy_class.predict_action_chunk).parameters
        accepts_rtc_kwargs = {"inference_delay", "prev_chunk_left_over"}.issubset(
            predict_chunk_params
        ) or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in predict_chunk_params.values())
        if not accepts_rtc_kwargs:
            raise ValueError(
                f"Policy type '{policy_cfg.type}' does not support RTC inference: "
                f"predict_action_chunk() has no inference_delay/prev_chunk_left_over/**kwargs. "
                f"RTC is only supported by VLA-style policies (pi0, pi05, pi0_fast, smolvla, evo1, "
                f"molmoact2). Use --inference.type=sync instead."
            )
        rtc_config = RTCConfig(
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
        )
        policy.config.rtc_config = rtc_config
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()
        inference_cfg = RTCInferenceConfig(rtc=rtc_config, queue_threshold=args.rtc_queue_threshold)
    else:
        inference_cfg = SyncInferenceConfig()

    policy = policy.to(device)
    policy.eval()

    # === Robot ===
    robot_config = AlohaMiniClientConfig(
        remote_ip=args.remote_ip, id=args.robot_id, robot_model=args.robot_model
    )
    robot = AlohaMiniClient(robot_config)
    robot.connect()
    robot_wrapper = ThreadSafeRobot(robot)

    # === Processors ===
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # === Dataset features ===
    # Use all observation features (pos + base vel/height) to match what record_bi.py records.
    observation_features_hw = robot.observation_features
    action_features_hw = robot.action_features

    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=True,
    )
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=True,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")
    ordered_action_keys = list(action_features_hw.keys())

    # === Dataset ===
    dataset = None
    if args.record_dataset:
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_repo_id,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

    # === Policy processors (needs dataset stats) ===
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        dataset_stats=dataset.meta.stats if dataset is not None else None,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # === Inference engine ===
    engine = create_inference_engine(
        inference_cfg,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        hw_features=hw_features,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=args.task_description,
        fps=float(args.fps),
        device=device,
    )
    engine.start()
    engine.resume()

    # init_rerun(session_name="alohamini_evaluate")
    log_say("Starting evaluation")

    interpolator = ActionInterpolator(multiplier=args.interpolation_multiplier)
    control_interval = interpolator.get_control_interval(args.fps)
    recorded = 0

    while recorded < args.num_episodes:
        log_say(f"Eval episode {recorded + 1} of {args.num_episodes}")
        engine.reset()
        interpolator.reset()
        start = time.perf_counter()
        cached_obs_processed = None

        while (time.perf_counter() - start) < args.episode_time:
            loop_start = time.perf_counter()

            obs_raw = robot.get_observation()
            if cached_obs_processed is None or interpolator.needs_new_action():
                obs_processed = robot_observation_processor(obs_raw)
                engine.notify_observation(obs_processed)
                cached_obs_processed = obs_processed
            else:
                obs_processed = cached_obs_processed
            obs_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            if interpolator.needs_new_action():
                action_tensor = engine.get_action(obs_frame)
                if action_tensor is not None:
                    interpolator.add(action_tensor.cpu())

            interp_action = interpolator.get()
            if interp_action is not None:
                action_dict = {k: interp_action[i].item() for i, k in enumerate(ordered_action_keys)}
                action_dict = snap_planar_velocity(
                    action_dict,
                    speed=args.base_snap_speed,
                    deadband=args.base_snap_deadband,
                )
                robot.send_action(robot_action_processor((action_dict, obs_raw)))
                if dataset is not None:
                    action_frame = build_dataset_frame(dataset_features, action_dict, prefix=ACTION)
                    dataset.add_frame({**obs_frame, **action_frame, "task": args.task_description})

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)

        if dataset is not None:
            dataset.save_episode()
        recorded += 1

    log_say("Evaluation complete")
    engine.stop()
    robot.disconnect()
    if dataset is not None:
        dataset.finalize()
        if args.push_to_hub:
            dataset.push_to_hub()


if __name__ == "__main__":
    main()
