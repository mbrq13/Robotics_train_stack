import json

import numpy as np
import pytest

from robotics_stack.guidance.authority import GuidancePhase
from robotics_stack.guidance.records import GuidedEpisode, GuidedSample


def _sample(
    observation_id: int,
    phase: GuidancePhase,
    generation: int,
) -> GuidedSample:
    return GuidedSample(
        observation_id=observation_id,
        station_monotonic_ns=observation_id * 1_000,
        state=(0.1, 0.2),
        action=(0.3, 0.4),
        phase=phase,
        control_generation=generation,
    )


def test_guided_episode_exports_training_arrays_and_provenance(tmp_path) -> None:
    episode = GuidedEpisode(task="place", schema_version=1, episode_id="episode-1")
    episode.append(_sample(1, GuidancePhase.POLICY_RUN, 0))
    episode.append(_sample(2, GuidancePhase.RECOVERY, 1))
    episode.append(_sample(3, GuidancePhase.CORRECTION, 2))

    output = episode.export_npz(tmp_path / "episode.npz")
    dataset = np.load(output)
    assert dataset["states"].shape == (3, 2)
    assert dataset["operator_mask"].tolist() == [False, True, True]
    assert dataset["control_generation"].tolist() == [0, 1, 2]
    assert json.loads(output.with_suffix(".json").read_text()) == {
        "episode_id": "episode-1",
        "task": "place",
        "schema_version": 1,
        "samples": 3,
        "operator_samples": 2,
        "policy_samples": 1,
    }


def test_guided_episode_rejects_out_of_order_samples() -> None:
    episode = GuidedEpisode(task="place", schema_version=1)
    episode.append(_sample(2, GuidancePhase.POLICY_RUN, 1))

    with pytest.raises(ValueError, match="increasing observation ids"):
        episode.append(_sample(2, GuidancePhase.CORRECTION, 1))
