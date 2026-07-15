from __future__ import annotations

import numpy as np
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.config.schema import EncoderConfig, ModelConfig, SequenceModelConfig
from gk_surrogate.data.factory import build_dataset
from gk_surrogate.data.normalization import (
    NormalizationStats,
    _RunningMoments,
    estimate_dataset_stats,
    estimate_trajectory_stats,
    normalize_snapshot,
)
from gk_surrogate.factory import build_encoder, build_sequence_model, build_simsiam_encoder_with_diagnostics
from gk_surrogate.models.encoders import ConvNDEncoder, PatchTransformerEncoder
from gk_surrogate.models.sequence import (
    CausalTransformerSequenceModel,
    GRUSequenceModel,
    GuppyLatentTransformer,
    MLPDeltaSequenceModel,
)


def test_normalization_modes_and_stats(tmp_path, tiny_config_path):
    x = np.arange(2 * 4 * 4 * 4 * 4 * 4, dtype=np.float32).reshape(2, 4, 4, 4, 4, 4)
    assert np.allclose(normalize_snapshot(x, mode="none"), x)
    assert abs(float(normalize_snapshot(x, mode="sample").mean())) < 1e-5
    stats = NormalizationStats(mean=np.asarray(1.0, dtype=np.float32), std=np.asarray(2.0, dtype=np.float32))
    assert normalize_snapshot(x, mode="fixed", stats=stats).shape == x.shape
    channel_stats = NormalizationStats(
        mean=np.asarray([0.0, 100.0], dtype=np.float32),
        std=np.asarray([1.0, 2.0], dtype=np.float32),
    )
    channel_x = np.stack([np.full((2, 2, 2, 2, 2), 1.0), np.full((2, 2, 2, 2, 2), 102.0)]).astype(np.float32)
    channel_normalized = normalize_snapshot(channel_x, mode="fixed", stats=channel_stats)
    assert np.allclose(channel_normalized[0], 1.0)
    assert np.allclose(channel_normalized[1], 1.0)
    with pytest.raises(ValueError, match="does not match 2 channels"):
        normalize_snapshot(
            channel_x,
            mode="fixed",
            stats=NormalizationStats(mean=np.asarray([0.0]), std=np.asarray([1.0])),
        )
    stats.save_npz(tmp_path / "stats.npz")
    assert (tmp_path / "stats.npz").exists()
    with pytest.raises(ValueError):
        normalize_snapshot(x, mode="fixed")
    config = load_config(tiny_config_path, command="train-encoder")
    dataset = build_dataset(config.data)
    estimated = estimate_dataset_stats(dataset, max_samples=2)
    assert float(estimated.std) > 0
    trajectory_stats = estimate_trajectory_stats(dataset, dataset.trajectory_ids()[0])
    assert float(trajectory_stats.std) > 0

    class EmptyDataset:
        def trajectory_ids(self):
            return ()

    with pytest.raises(ValueError, match="empty"):
        estimate_dataset_stats(EmptyDataset())

    class EmptyTrajectoryDataset:
        def num_timesteps(self, trajectory_id):
            return 0

    with pytest.raises(ValueError, match="empty trajectory"):
        estimate_trajectory_stats(EmptyTrajectoryDataset(), "empty")


def test_dataset_normalization_includes_between_snapshot_variance():
    class ShiftedDataset:
        def trajectory_ids(self):
            return ("trajectory",)

        def num_timesteps(self, trajectory_id):
            assert trajectory_id == "trajectory"
            return 2

        def get_snapshot(self, trajectory_id, timestep):
            assert trajectory_id == "trajectory"
            value = float(timestep * 10)
            return type("Sample", (), {"x": np.full((1, 2, 2, 2, 2, 2), value, dtype=np.float32)})()

    stats = estimate_dataset_stats(ShiftedDataset(), max_samples=2)
    assert float(stats.mean) == pytest.approx(5.0)
    assert float(stats.std) == pytest.approx(5.0)

    trajectory_stats = estimate_trajectory_stats(ShiftedDataset(), "trajectory")
    assert float(trajectory_stats.mean) == pytest.approx(5.0)
    assert float(trajectory_stats.std) == pytest.approx(5.0)


def test_dataset_normalization_can_be_restricted_to_split_trajectories():
    class SplitDataset:
        def trajectory_ids(self):
            return ("train", "held_out")

        def num_timesteps(self, trajectory_id):
            return 2

        def get_snapshot(self, trajectory_id, timestep):
            del timestep
            value = 1.0 if trajectory_id == "train" else 1_000.0
            return type("Sample", (), {"x": np.full((4,), value, dtype=np.float32)})()

    stats = estimate_dataset_stats(SplitDataset(), trajectory_ids=("train",))
    assert float(stats.mean) == pytest.approx(1.0)
    assert float(stats.std) == pytest.approx(1e-6)


def test_dataset_normalization_is_stable_for_large_offsets():
    class LargeOffsetDataset:
        def trajectory_ids(self):
            return ("trajectory",)

        def num_timesteps(self, trajectory_id):
            assert trajectory_id == "trajectory"
            return 2

        def get_snapshot(self, trajectory_id, timestep):
            assert trajectory_id == "trajectory"
            value = np.float32(100_000_000 + timestep * 16)
            return type("Sample", (), {"x": np.full((8,), value, dtype=np.float32)})()

    stats = estimate_dataset_stats(LargeOffsetDataset(), max_samples=2)
    assert float(stats.mean) == pytest.approx(100_000_008)
    assert float(stats.std) == pytest.approx(8.0)


def test_normalization_rejects_invalid_parameters_and_stats_shapes():
    x = np.ones((2, 2, 2), dtype=np.float32)
    scalar_stats = NormalizationStats(np.asarray(1.0), np.asarray(2.0))

    with pytest.raises(ValueError, match="epsilon must be positive"):
        normalize_snapshot(x, mode="none", epsilon=0)
    for mode in ("trajectory", "dataset"):
        with pytest.raises(ValueError, match="requires precomputed stats"):
            normalize_snapshot(x, mode=mode)
        normalized = normalize_snapshot(x, mode=mode, stats=scalar_stats)
        np.testing.assert_allclose(normalized, 0.0)
    with pytest.raises(ValueError, match="unknown normalization mode"):
        normalize_snapshot(x, mode="global")
    with pytest.raises(ValueError, match="do not broadcast"):
        normalize_snapshot(
            x,
            mode="fixed",
            stats=NormalizationStats(np.ones((3, 4)), np.ones((3, 4))),
        )

    class OneSnapshotDataset:
        def trajectory_ids(self):
            return ("trajectory",)

        def num_timesteps(self, trajectory_id):
            return 1

        def get_snapshot(self, trajectory_id, timestep):
            return type("Sample", (), {"x": x})()

    dataset = OneSnapshotDataset()
    with pytest.raises(ValueError, match="max_samples must be positive"):
        estimate_dataset_stats(dataset, max_samples=0)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        estimate_dataset_stats(dataset, epsilon=0)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        estimate_trajectory_stats(dataset, "trajectory", epsilon=0)

    moments = _RunningMoments()
    moments.update(np.asarray([], dtype=np.float32))
    assert moments.count == 0
    with pytest.raises(ValueError, match="without values"):
        moments.finalize(epsilon=1e-6)


def test_factory_builds_encoder_and_sequence_variants():
    conv = build_encoder(
        EncoderConfig(
            type="conv_nd",
            latent_dim=8,
            hidden_dims=(4,),
            extra={"kernel_size": [3, 3, 3, 3, 3], "strides": [[1, 1, 1, 1, 1]]},
        )
    )
    patch = build_encoder(
        EncoderConfig(
            type="patch_transformer",
            latent_dim=8,
            extra={"patch_size": [2, 2, 2, 2, 2], "embed_dim": 8, "depth": 1, "num_heads": 2},
        )
    )
    assert isinstance(conv, ConvNDEncoder)
    assert isinstance(patch, PatchTransformerEncoder)
    assert isinstance(build_sequence_model(SequenceModelConfig(type="mlp_delta", latent_dim=8)), MLPDeltaSequenceModel)
    assert isinstance(build_sequence_model(SequenceModelConfig(type="gru", latent_dim=8)), GRUSequenceModel)
    assert isinstance(
        build_sequence_model(
            SequenceModelConfig(type="causal_transformer", latent_dim=8, extra={"embed_dim": 8, "num_heads": 2})
        ),
        CausalTransformerSequenceModel,
    )
    assert isinstance(
        build_sequence_model(
            SequenceModelConfig(
                type="guppy_latent_transformer",
                latent_dim=8,
                extra={"model_dim": 8, "num_heads": 2, "predict_delta": True},
            )
        ),
        GuppyLatentTransformer,
    )
    assert isinstance(
        build_sequence_model(
            SequenceModelConfig(type="gpt_latent_transformer", latent_dim=8, extra={"model_dim": 8, "num_heads": 2})
        ),
        GuppyLatentTransformer,
    )
    with pytest.raises(ValueError):
        build_encoder(EncoderConfig(type="unknown", latent_dim=8))


def test_factory_builds_remaining_adapters_and_guards():
    external = build_encoder(EncoderConfig(type="external_adapter", latent_dim=4, extra={"name": "future"}))
    assert external.name == "future"
    with pytest.raises(ValueError, match="model.simsiam"):
        build_simsiam_encoder_with_diagnostics(ModelConfig(simsiam=None))
    assert build_sequence_model(SequenceModelConfig(type="persistence", latent_dim=4)).latent_dim == 4
    assert build_sequence_model(SequenceModelConfig(type="gpt2_adapter", latent_dim=4)).latent_dim == 4
