# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import inspect
import logging
import math
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import LatencyTracker, reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()
        self.observations_received = 0
        self.observations_enqueued = 0
        self.observations_filtered = 0
        self.observations_replaced = 0

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.rename_map = {}
        self.inference_type = "async"
        self.rtc_config = None
        self.rtc_latency_tracker = LatencyTracker()
        self.rtc_original_chunk = None
        self.rtc_processed_chunk = None
        self.rtc_chunk_timestep = 0
        self.rtc_relative_step = None
        self.rtc_normalizer_step = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)
        self.last_processed_obs = None

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
        self.observations_received = 0
        self.observations_enqueued = 0
        self.observations_filtered = 0
        self.observations_replaced = 0
        self.rtc_latency_tracker.reset()
        self.rtc_original_chunk = None
        self.rtc_processed_chunk = None
        self.rtc_chunk_timestep = 0

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.rename_map = policy_specs.rename_map
        self.inference_type = policy_specs.inference_type

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        if self.inference_type == "rtc":
            predict_chunk_params = inspect.signature(policy_class.predict_action_chunk).parameters
            accepts_rtc_kwargs = {"inference_delay", "prev_chunk_left_over"}.issubset(
                predict_chunk_params
            ) or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in predict_chunk_params.values())
            if not accepts_rtc_kwargs:
                raise ValueError(
                    f"Policy type {self.policy_type!r} does not support RTC: predict_action_chunk() "
                    "must accept inference_delay and prev_chunk_left_over"
                )
            self.rtc_config = RTCConfig(
                execution_horizon=policy_specs.rtc_execution_horizon,
                max_guidance_weight=policy_specs.rtc_max_guidance_weight,
            )
            self.policy.config.rtc_config = self.rtc_config
            if hasattr(self.policy, "init_rtc_processor"):
                self.policy.init_rtc_processor()
        self.policy.to(self.device)
        self.policy.eval()

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        preprocessor_overrides = {"device_processor": device_override}
        if policy_specs.rename_map:
            preprocessor_overrides["rename_observations_processor"] = {"rename_map": policy_specs.rename_map}

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": device_override},
        )
        if self.inference_type == "rtc":
            self.rtc_relative_step = next(
                (
                    step
                    for step in self.preprocessor.steps
                    if isinstance(step, RelativeActionsProcessorStep) and step.enabled
                ),
                None,
            )
            self.rtc_normalizer_step = next(
                (step for step in self.preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
                None,
            )
            if self.rtc_relative_step is not None and self.rtc_relative_step.action_names is None:
                action_names = getattr(self.policy.config, "action_feature_names", None)
                if action_names:
                    self.rtc_relative_step.action_names = list(action_names)

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        self.observations_received += 1
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        enqueued = self._enqueue_observation(timed_observation)  # wrapping a RawObservation
        if not enqueued:
            self.observations_filtered += 1
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")
        if self.observations_received % 50 == 0:
            self.logger.info(
                "Observation flow | received=%d enqueued=%d filtered=%d replaced=%d",
                self.observations_received,
                self.observations_enqueued,
                self.observations_filtered,
                self.observations_replaced,
            )

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go}, "
                f"age_ms={(time.time() - obs.get_timestamp()) * 1000:.1f})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        # Joint/velocity state can remain constant while a mobile robot moves through a changing
        # scene. Never use the legacy state-only similarity filter for visual policies or explicit
        # closed-loop modes. The maxsize=1 queue already bounds backlog and keeps only the newest.
        has_visual_inputs = self.policy is not None and bool(self.policy_image_features)
        should_enqueue = (
            has_visual_inputs
            or self.inference_type in {"rtc", "sync"}
            or (
                obs.must_go
                or self.last_processed_obs is None
                or self._obs_sanity_checks(obs, self.last_processed_obs)
            )
        )
        if should_enqueue:
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.observations_replaced += 1
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            self.observations_enqueued += 1
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_rtc_left_over(self, timestep: int) -> torch.Tensor | None:
        """Return the unexecuted prefix from the previous *model-space* chunk."""
        if self.rtc_original_chunk is None:
            return None
        consumed = max(0, timestep - self.rtc_chunk_timestep)
        left_over = self.rtc_original_chunk[consumed:]
        if self.rtc_relative_step is not None and self.rtc_processed_chunk is not None:
            raw_state = self.rtc_relative_step.get_cached_state()
            processed_left_over = self.rtc_processed_chunk[consumed:]
            if raw_state is not None and processed_left_over.numel() > 0:
                left_over = reanchor_relative_rtc_prefix(
                    prev_actions_absolute=processed_left_over,
                    current_state=raw_state,
                    relative_step=self.rtc_relative_step,
                    normalizer_step=self.rtc_normalizer_step,
                    policy_device=torch.device(self.device),
                )
        horizon = self.rtc_config.execution_horizon
        if len(left_over) >= horizon:
            return left_over[:horizon]
        padded = torch.zeros((horizon, left_over.shape[-1]), dtype=left_over.dtype, device=left_over.device)
        padded[: len(left_over)] = left_over
        return padded

    def _get_action_chunk(
        self, observation: dict[str, torch.Tensor], observation_timestep: int | None = None
    ) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        if self.inference_type == "sync":
            chunk = self.policy.select_action(observation).unsqueeze(1)
        elif self.inference_type == "rtc":
            latency = self.rtc_latency_tracker.max()
            inference_delay = math.ceil(latency / self.config.environment_dt) if latency else 0
            prev_chunk_left_over = self._get_rtc_left_over(observation_timestep or 0)
            chunk = self.policy.predict_action_chunk(
                observation,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
            )
        else:
            chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            rename_map=self.rename_map,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation, observation_t.get_timestep())
        inference_time = time.perf_counter() - start_inference
        if self.inference_type == "rtc":
            self.rtc_latency_tracker.add(inference_time)
            self.rtc_original_chunk = action_tensor.squeeze(0).detach().clone()
            self.rtc_chunk_timestep = observation_t.get_timestep()
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        if self.inference_type == "rtc":
            self.rtc_processed_chunk = action_tensor.detach().clone()
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
