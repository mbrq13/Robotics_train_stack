"""Checkpoint inspection and loading for deployable policies.

The deployment boundary intentionally supports two artifact families: native
artifacts produced by this repository and standard LeRobot Pi0.5 repositories.
An external checkpoint is never treated as compatible merely because it loads:
its declared inputs and action order must match the connected station first.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robotics_stack.contracts import CheckpointManifest, ContractError, Observation, RobotSchema
from robotics_stack.policy.mlp import StateMlpPolicy
from robotics_stack.policy.rtc import RtcChunk, RtcSettings

_NESTED_ARTIFACT = Path("checkpoints/best_mean/pretrained_model")


def _checkpoint_file(checkpoint: str | Path, filename: str) -> Path:
    root = Path(checkpoint)
    if root.is_dir():
        candidate = root / filename
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"{filename} is missing from {root}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "install the policy-lerobot extra to inspect a Hugging Face checkpoint"
        ) from exc
    return Path(hf_hub_download(repo_id=str(checkpoint), filename=filename))


def _policy_config_path(checkpoint: str | Path) -> tuple[Path, Path]:
    """Locate a policy config at a repository root or a saved best checkpoint."""
    failures: list[Exception] = []
    for relative in (Path("config.json"), _NESTED_ARTIFACT / "config.json"):
        try:
            return _checkpoint_file(checkpoint, str(relative)), relative.parent
        except (FileNotFoundError, OSError) as exc:
            failures.append(exc)
    raise FileNotFoundError(f"no deployable policy config found in {checkpoint}") from failures[-1]


def _materialize_policy_root(checkpoint: str | Path) -> Path:
    """Return a local artifact directory, downloading only when a Hub id is used."""
    config_path, relative_root = _policy_config_path(checkpoint)
    root = Path(checkpoint)
    if root.is_dir():
        return config_path.parent
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub to load a Hugging Face checkpoint") from exc
    prefix = "" if relative_root == Path(".") else f"{relative_root.as_posix()}/"
    cached_root = Path(
        snapshot_download(repo_id=str(checkpoint), allow_patterns=[f"{prefix}**"])
    )
    return cached_root / relative_root


def _feature_shape(feature: Any) -> tuple[int, ...]:
    if isinstance(feature, dict):
        shape = feature.get("shape", ())
    else:
        shape = ()
    return tuple(int(value) for value in shape)


def _canonical_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


@dataclass(frozen=True)
class CheckpointDescriptor:
    """Public, framework-neutral requirements extracted from a checkpoint."""

    kind: str
    framework: str
    state_dim: int
    action_dim: int
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    cameras: dict[str, tuple[int, ...]]
    source: str
    rtc_training_max_delay: int = 0
    chunk_size: int = 0
    relative_actions: bool = False
    relative_exclude_joints: tuple[str, ...] = ()

    def validate_station(self, schema: RobotSchema) -> None:
        failures: list[str] = []
        if self.state_dim != len(schema.state_names):
            failures.append(
                f"state dimension checkpoint={self.state_dim}, station={len(schema.state_names)}"
            )
        if self.action_dim != schema.action_size:
            failures.append(
                f"action dimension checkpoint={self.action_dim}, station={schema.action_size}"
            )
        if self.kind == "pi05" and not self.state_names:
            failures.append("Pi0.5 state order is not declared by checkpoint or policy profile")
        if self.state_names and tuple(map(_canonical_name, self.state_names)) != tuple(
            map(_canonical_name, schema.state_names)
        ):
            failures.append("state order differs")
        if self.action_names and tuple(map(_canonical_name, self.action_names)) != tuple(
            map(_canonical_name, schema.action_names)
        ):
            failures.append("action order differs")
        missing_cameras = sorted(set(self.cameras) - set(schema.camera_names))
        if missing_cameras:
            failures.append(f"station is missing cameras: {', '.join(missing_cameras)}")
        if failures:
            raise ContractError("checkpoint is incompatible with station: " + "; ".join(failures))


def inspect_checkpoint(
    checkpoint: str | Path, *, state_names: tuple[str, ...] = ()
) -> CheckpointDescriptor:
    """Read a descriptor without downloading model weights or constructing a model."""
    root = Path(checkpoint)
    if root.is_dir() and (root / "manifest.json").is_file():
        manifest = CheckpointManifest.from_dict(
            json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        )
        return CheckpointDescriptor(
            kind=manifest.policy_kind,
            framework=manifest.framework,
            state_dim=manifest.state_dim,
            action_dim=manifest.action_dim,
            state_names=manifest.schema.state_names,
            action_names=manifest.schema.action_names,
            cameras={name: () for name in manifest.schema.camera_names},
            source=str(root),
        )

    config_path, _ = _policy_config_path(checkpoint)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type") != "pi05":
        raise ValueError(f"unsupported external policy type: {config.get('type')!r}")
    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    state_shape = _feature_shape(inputs.get("observation.state"))
    action_shape = _feature_shape(outputs.get("action"))
    if len(state_shape) != 1 or len(action_shape) != 1:
        raise ValueError("Pi0.5 checkpoint must declare one-dimensional state and action features")
    cameras = {
        name.removeprefix("observation.images."): _feature_shape(feature)
        for name, feature in inputs.items()
        if name.startswith("observation.images.")
    }
    return CheckpointDescriptor(
        kind="pi05",
        framework="lerobot",
        state_dim=state_shape[0],
        action_dim=action_shape[0],
        # Pi0.5 config.json commonly does not encode state feature names.  The
        # deployment profile must supply them in that case; never guess a state
        # vector order from its dimensionality.
        state_names=tuple(config.get("state_feature_names", ())) or state_names,
        action_names=tuple(config.get("action_feature_names", ())),
        cameras=cameras,
        source=str(checkpoint),
        rtc_training_max_delay=int(config.get("rtc_training_max_delay", 0) or 0),
        chunk_size=int(config.get("chunk_size", 0) or 0),
        relative_actions=bool(config.get("use_relative_actions", False)),
        relative_exclude_joints=tuple(config.get("relative_exclude_joints", ())),
    )


class DeployedPolicy(Protocol):
    descriptor: CheckpointDescriptor

    def predict(self, observation: Observation) -> tuple[float, ...]: ...

    def configure_rtc(self, settings: RtcSettings) -> None: ...

    def predict_rtc_chunk(
        self,
        observation: Observation,
        *,
        inference_delay_steps: int,
        previous_raw_actions: np.ndarray | None,
        previous_actions: np.ndarray | None = None,
    ) -> RtcChunk: ...


class NativePolicy:
    def __init__(self, policy: StateMlpPolicy, manifest: CheckpointManifest):
        self.policy = policy
        self.descriptor = CheckpointDescriptor(
            kind=manifest.policy_kind,
            framework=manifest.framework,
            state_dim=manifest.state_dim,
            action_dim=manifest.action_dim,
            state_names=manifest.schema.state_names,
            action_names=manifest.schema.action_names,
            cameras={name: () for name in manifest.schema.camera_names},
            source="native artifact",
        )

    def predict(self, observation: Observation) -> tuple[float, ...]:
        return self.policy.predict(observation.state)

    def configure_rtc(self, settings: RtcSettings) -> None:
        del settings
        raise ValueError("native state_mlp artifacts do not implement chunked RTC inference")

    def predict_rtc_chunk(
        self,
        observation: Observation,
        *,
        inference_delay_steps: int,
        previous_raw_actions: np.ndarray | None,
        previous_actions: np.ndarray | None = None,
    ) -> RtcChunk:
        del observation, inference_delay_steps, previous_raw_actions, previous_actions
        raise ValueError("native state_mlp artifacts do not implement chunked RTC inference")


class LeRobotPi05Policy:
    """Thin adapter around the public LeRobot Pi0.5 inference interface."""

    def __init__(
        self,
        checkpoint: str | Path,
        task: str,
        device: str,
        state_names: tuple[str, ...] = (),
        robot_type: str = "",
    ):
        self.checkpoint = str(_materialize_policy_root(checkpoint))
        self.task = task
        self.robot_type = robot_type
        self.descriptor = inspect_checkpoint(checkpoint, state_names=state_names)
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
            from lerobot.policies.utils import prepare_observation_for_inference
        except ImportError as exc:
            raise RuntimeError(
                "install the policy-lerobot extra in the worker environment"
            ) from exc

        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        if config.type != "pi05":
            raise ValueError(f"expected pi05 checkpoint, got {config.type!r}")
        config.device = device
        self._torch = torch
        self._config = config
        self._device = torch.device(device)
        self._prepare_observation = prepare_observation_for_inference
        policy_class = get_policy_class(config.type)
        self._policy = policy_class.from_pretrained(self.checkpoint, config=config)
        self._policy.to(self._device)
        self._policy.eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=self.checkpoint,
            dataset_stats={},
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self._use_amp = bool(getattr(config, "use_amp", False))
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    @staticmethod
    def _as_array(value: Any) -> np.ndarray:
        if isinstance(value, dict):
            value = value.get("action")
        elif hasattr(value, "action"):
            value = value.action
        if value is None:
            raise ContractError("Pi0.5 postprocessor did not return an action")
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _frame(self, observation: Observation) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "install the policy-lerobot extra in the worker environment"
            ) from exc
        frame: dict[str, Any] = {
            "observation.state": np.asarray(observation.state, dtype=np.float32),
        }
        for name, expected_shape in self.descriptor.cameras.items():
            encoded = observation.images.get(name)
            if encoded is None:
                raise ContractError(f"observation is missing image {name!r}")
            image = np.asarray(Image.open(BytesIO(encoded)).convert("RGB"))
            actual_shape = (image.shape[2], image.shape[0], image.shape[1])
            if expected_shape and actual_shape != expected_shape:
                raise ContractError(
                    f"camera {name} shape is {actual_shape}; checkpoint requires {expected_shape}"
                )
            frame[f"observation.images.{name}"] = image
        return frame

    def predict(self, observation: Observation) -> tuple[float, ...]:
        frame = self._frame(observation)
        amp = (
            self._torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._use_amp
            else nullcontext()
        )
        with self._torch.inference_mode(), amp:
            batch = self._prepare_observation(frame, self._device, self.task, self.robot_type)
            action = self._postprocessor(self._policy.select_action(self._preprocessor(batch)))
        values = self._as_array(action).reshape(-1)
        if values.size != self.descriptor.action_dim or not np.isfinite(values).all():
            raise ContractError("Pi0.5 produced an invalid action")
        return tuple(float(value) for value in values)

    def configure_rtc(self, settings: RtcSettings) -> None:
        """Install the checkpoint-derived RTC runtime config before inference."""
        if self.descriptor.rtc_training_max_delay <= 0:
            raise ValueError(
                "trained RTC requires a checkpoint with rtc_training_max_delay > 0; "
                "run normal deployment or train the policy with RTC enabled"
            )
        if settings.training_max_delay != self.descriptor.rtc_training_max_delay:
            raise ValueError("RTC runtime delay must equal the checkpoint training delay")
        settings.validate_chunk_size(self.descriptor.chunk_size)
        try:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
        except ImportError as exc:
            raise RuntimeError("the installed LeRobot version does not provide RTC") from exc

        # Pi0.5 checkpoints such as hanoi-v1 may carry the training delay but
        # omit rtc_config.  Reconstruct the persisted runtime metadata from
        # that explicit training contract instead of guessing a different mode.
        try:
            rtc_config = RTCConfig(
                enabled=True,
                mode="trained",
                execution_horizon=settings.execution_horizon,
            )
        except TypeError:  # Compatibility with early LeRobot RTC releases.
            rtc_config = RTCConfig(mode="trained", execution_horizon=settings.execution_horizon)
            if hasattr(rtc_config, "enabled"):
                rtc_config.enabled = True
        for target in (self._config, getattr(self._policy, "config", None)):
            if target is not None:
                target.rtc_config = rtc_config
        model = getattr(self._policy, "model", None)
        if model is not None and hasattr(model, "config"):
            model.config.rtc_config = rtc_config
        initializer = getattr(self._policy, "init_rtc_processor", None)
        if initializer is None:
            raise RuntimeError("the loaded Pi0.5 implementation does not support RTC inference")
        initializer()

    def predict_rtc_chunk(
        self,
        observation: Observation,
        *,
        inference_delay_steps: int,
        previous_raw_actions: np.ndarray | None,
        previous_actions: np.ndarray | None = None,
    ) -> RtcChunk:
        frame = self._frame(observation)
        amp = (
            self._torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._use_amp
            else nullcontext()
        )
        previous = (
            None
            if previous_raw_actions is None
            else self._torch.as_tensor(
                previous_raw_actions,
                device=self._device,
                dtype=self._torch.float32,
            )
        )
        with self._torch.inference_mode(), amp:
            batch = self._prepare_observation(frame, self._device, self.task, self.robot_type)
            processed = self._preprocessor(batch)
            if self.descriptor.relative_actions and previous_actions is not None:
                previous = self._reanchor_relative_prefix(previous_actions)
            raw_chunk = self._policy.predict_action_chunk(
                processed,
                inference_delay=max(0, int(inference_delay_steps)),
                prev_chunk_left_over=previous,
            )
            actions = self._postprocessor(raw_chunk)
        raw = self._as_array(raw_chunk)
        output = self._as_array(actions)
        if raw.ndim == 3:
            raw = raw[0]
        if output.ndim == 3:
            output = output[0]
        if raw.ndim != 2 or raw.shape[1] != self.descriptor.action_dim:
            raise ContractError("Pi0.5 RTC returned an invalid action chunk")
        return RtcChunk(raw=raw, actions=output)

    def _reanchor_relative_prefix(self, previous_actions: np.ndarray) -> Any:
        """Express absolute queued targets in the current relative-action frame."""
        try:
            from lerobot.policies.rtc import reanchor_relative_rtc_prefix
            from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep
        except ImportError as exc:
            raise RuntimeError(
                "the installed LeRobot version lacks relative-action RTC support"
            ) from exc
        relative_step = next(
            (
                step
                for step in self._preprocessor.steps
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        if relative_step is None:
            raise RuntimeError("relative checkpoint is missing its relative-action processor")
        current_state = relative_step.get_cached_state()
        if current_state is None:
            raise RuntimeError("relative-action processor did not retain the current state")
        normalizer_step = next(
            (
                step
                for step in self._preprocessor.steps
                if isinstance(step, NormalizerProcessorStep)
            ),
            None,
        )
        absolute = self._torch.as_tensor(
            previous_actions, device=self._device, dtype=self._torch.float32
        )
        return reanchor_relative_rtc_prefix(
            prev_actions_absolute=absolute,
            current_state=current_state,
            relative_step=relative_step,
            normalizer_step=normalizer_step,
            policy_device=self._device,
        )


def load_deployed_policy(
    checkpoint: str | Path,
    *,
    task: str = "",
    device: str = "cuda",
    state_names: tuple[str, ...] = (),
    robot_type: str = "",
) -> DeployedPolicy:
    """Load a native artifact or a standard LeRobot Pi0.5 checkpoint."""
    root = Path(checkpoint)
    if root.is_dir() and (root / "manifest.json").is_file():
        policy, manifest = StateMlpPolicy.load(root)
        return NativePolicy(policy, manifest)
    return LeRobotPi05Policy(checkpoint, task, device, state_names, robot_type)
