from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import pytest
import yaml

from gk_surrogate import pipeline
from gk_surrogate.config.load import load_config
from gk_surrogate.data.latent_cache import LatentCacheWriter
from gk_surrogate.evaluation import rollout as eval_rollout
from gk_surrogate.models.encoders import (
    ConvNDEncoder,
    FlattenMLPEncoder,
    PatchTransformerEncoder,
    _activation,
    _as_tuple,
)
from gk_surrogate.training import train_encoder as encoder_train
from gk_surrogate.training import train_sequence as sequence_train


def _snapshot_batch():
    return jnp.ones((2, 2, 4, 4, 4, 4, 4), dtype=jnp.float32)


def _latent_context():
    return jnp.ones((2, 3, 4), dtype=jnp.float32)


def _write_protocol_checkpoint(tmp_path, trajectory_ids, *, seed=52, role="encoder_checkpoint"):
    run_dir = tmp_path / role
    checkpoint = run_dir / "checkpoints" / "step_000001"
    checkpoint.mkdir(parents=True)
    train_ids = pipeline.split_trajectory_ids(trajectory_ids, seed=seed)["train"]
    config = {
        "data": {"backend": "cyclone_kvikio", "seed": seed, "split": "train"},
        "training": {"seed": seed + 1},
    }
    metrics = {
        "protocol_version": 1,
        "artifact_role": role,
        "data_backend": "cyclone_kvikio",
        "data_split": "train",
        "data_split_seed": seed,
        "training_seed": seed + 1,
        "selected_trajectory_ids": list(train_ids),
        "trajectory_manifest_sha256": pipeline._trajectory_manifest_sha256(train_ids),
        "universe_trajectory_ids": list(trajectory_ids),
        "universe_manifest_sha256": pipeline._trajectory_manifest_sha256(trajectory_ids),
        "checkpoint": str(checkpoint),
    }
    (run_dir / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return checkpoint, config, metrics


def test_pipeline_rejects_cross_seed_cache_and_checkpoint_provenance(tmp_path):
    cache_path = tmp_path / "latent_cache.h5"
    LatentCacheWriter(
        cache_path,
        latent_dim=2,
        config_yaml=yaml.safe_dump({"data": {"seed": 11}}),
    )
    with pytest.raises(ValueError, match="latent cache split seed 11"):
        pipeline._validate_latent_cache_split_seed(cache_path, expected_seed=12)

    checkpoint = tmp_path / "sequence" / "checkpoints" / "step_000001"
    checkpoint.mkdir(parents=True)
    (tmp_path / "sequence" / "config_resolved.json").write_text(
        json.dumps({"data": {"seed": 21}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sequence checkpoint split seed 21"):
        pipeline._validate_checkpoint_split_seed(
            checkpoint,
            expected_seed=22,
            role="sequence checkpoint",
        )


def test_real_checkpoint_protocol_is_fail_closed_and_manifest_bound(tmp_path):
    trajectory_ids = ("traj-b", "traj-a", "traj-c", "traj-d")
    checkpoint, config, metrics = _write_protocol_checkpoint(tmp_path, trajectory_ids)
    kwargs = {
        "expected_seed": 52,
        "expected_backend": "cyclone_kvikio",
        "expected_universe_ids": trajectory_ids,
        "expected_artifact_role": "encoder_checkpoint",
        "role": "encoder checkpoint",
        "require_complete": True,
    }
    pipeline._validate_checkpoint_protocol(checkpoint, **kwargs)

    metrics_path = checkpoint.parents[1] / "metrics.json"
    metrics_path.unlink()
    with pytest.raises(ValueError, match="missing colocated"):
        pipeline._validate_checkpoint_protocol(checkpoint, **kwargs)

    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    config["data"]["split"] = "all"
    (checkpoint.parents[1] / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="data.split='train'"):
        pipeline._validate_checkpoint_protocol(checkpoint, **kwargs)

    config["data"]["split"] = "train"
    (checkpoint.parents[1] / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    metrics["trajectory_manifest_sha256"] = "tampered"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="training trajectory manifest"):
        pipeline._validate_checkpoint_protocol(checkpoint, **kwargs)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_config_mapping", "missing data/training"),
        ("config_seed", "split seed"),
        ("config_backend", "does not match configured backend"),
        ("missing_metric", "missing protocol fields"),
        ("protocol_version", "unsupported protocol version"),
        ("artifact_role", "artifact role"),
        ("metrics_backend", "metrics backend"),
        ("metrics_split", "required seed-52 training split"),
        ("training_seed", "training seed differs"),
        ("universe_ids", "trajectory universe"),
        ("universe_manifest", "trajectory-universe manifest"),
        ("selected_ids", "canonical training split"),
        ("checkpoint_path", "does not identify the loaded checkpoint"),
    ],
)
def test_real_checkpoint_protocol_rejects_inconsistent_metadata(tmp_path, case, message):
    trajectory_ids = ("traj-0", "traj-1", "traj-2", "traj-3")
    checkpoint, config, metrics = _write_protocol_checkpoint(tmp_path, trajectory_ids)
    if case == "missing_config_mapping":
        config.pop("training")
    elif case == "config_seed":
        config["data"]["seed"] = 53
    elif case == "config_backend":
        config["data"]["backend"] = "h5"
    elif case == "missing_metric":
        metrics.pop("artifact_role")
    elif case == "protocol_version":
        metrics["protocol_version"] = 2
    elif case == "artifact_role":
        metrics["artifact_role"] = "sequence_checkpoint"
    elif case == "metrics_backend":
        metrics["data_backend"] = "h5"
    elif case == "metrics_split":
        metrics["data_split"] = "val"
    elif case == "training_seed":
        metrics["training_seed"] = 999
    elif case == "universe_ids":
        metrics["universe_trajectory_ids"] = list(reversed(trajectory_ids))
    elif case == "universe_manifest":
        metrics["universe_manifest_sha256"] = "tampered"
    elif case == "selected_ids":
        metrics["selected_trajectory_ids"] = [trajectory_ids[0]]
    elif case == "checkpoint_path":
        metrics["checkpoint"] = str(tmp_path / "different-checkpoint")
    (checkpoint.parents[1] / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    (checkpoint.parents[1] / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        pipeline._validate_checkpoint_protocol(
            checkpoint,
            expected_seed=52,
            expected_backend="cyclone_kvikio",
            expected_universe_ids=trajectory_ids,
            expected_artifact_role="encoder_checkpoint",
            role="encoder checkpoint",
            require_complete=True,
        )


def test_real_sequence_checkpoint_is_bound_to_cache_and_encoder_lineage(tmp_path):
    trajectory_ids = ("traj-0", "traj-1", "traj-2", "traj-3")
    checkpoint, config, metrics = _write_protocol_checkpoint(
        tmp_path,
        trajectory_ids,
        role="sequence_checkpoint",
    )
    cache_path = tmp_path / "cache.h5"
    encoder_checkpoint = tmp_path / "encoder" / "checkpoints" / "step_000001"
    cache_path.write_bytes(b"cache")
    encoder_checkpoint.mkdir(parents=True)
    (encoder_checkpoint / "checkpoint.pkl").write_bytes(b"encoder")
    config["latent_cache"] = {
        "path": str(cache_path),
        "encoder_checkpoint_path": str(encoder_checkpoint),
    }
    metrics["latent_cache"] = str(cache_path)
    metrics["encoder_checkpoint"] = str(encoder_checkpoint)
    metrics["latent_cache_sha256"] = pipeline._sha256_file(cache_path)
    metrics["encoder_checkpoint_sha256"] = pipeline._sha256_file(encoder_checkpoint / "checkpoint.pkl")
    (checkpoint.parents[1] / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    (checkpoint.parents[1] / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    kwargs = {
        "expected_seed": 52,
        "expected_backend": "cyclone_kvikio",
        "expected_universe_ids": trajectory_ids,
        "expected_artifact_role": "sequence_checkpoint",
        "role": "sequence checkpoint",
        "require_complete": True,
        "expected_cache_path": cache_path,
        "expected_encoder_checkpoint": encoder_checkpoint,
    }
    pipeline._validate_checkpoint_protocol(checkpoint, **kwargs)

    metrics["latent_cache"] = str(tmp_path / "different-cache.h5")
    (checkpoint.parents[1] / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics do not match"):
        pipeline._validate_checkpoint_protocol(checkpoint, **kwargs)

    metrics["latent_cache"] = str(cache_path)
    (checkpoint.parents[1] / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    config.pop("latent_cache")
    (checkpoint.parents[1] / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="missing latent-cache lineage"):
        pipeline._validate_checkpoint_protocol(checkpoint, **kwargs)

    config["latent_cache"] = {
        "path": str(cache_path),
        "encoder_checkpoint_path": str(encoder_checkpoint),
    }
    (checkpoint.parents[1] / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="requires cache and encoder lineage"):
        pipeline._validate_checkpoint_protocol(checkpoint, **{**kwargs, "expected_cache_path": None})

    config["latent_cache"]["path"] = str(tmp_path / "different-cache.h5")
    (checkpoint.parents[1] / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved config does not match"):
        pipeline._validate_checkpoint_protocol(checkpoint, **kwargs)


def test_real_latent_cache_protocol_matches_actual_universe_and_encoder(tmp_path):
    trajectory_ids = ("traj-b", "traj-a", "traj-c", "traj-d")
    checkpoint, _config, _metrics = _write_protocol_checkpoint(tmp_path, trajectory_ids)
    cache_config = {
        "data": {"backend": "cyclone_kvikio", "seed": 52, "split": "all"},
        "training": {"seed": 52},
        "latent_cache": {"encoder_checkpoint_path": str(checkpoint)},
    }
    protocol = {
        "protocol_version": 1,
        "artifact_role": "latent_cache",
        "data_backend": "cyclone_kvikio",
        "data_split": "all",
        "data_split_seed": 52,
        "training_seed": 52,
        "selected_trajectory_ids": list(trajectory_ids),
        "trajectory_manifest_sha256": pipeline._trajectory_manifest_sha256(trajectory_ids),
        "universe_trajectory_ids": list(trajectory_ids),
        "universe_manifest_sha256": pipeline._trajectory_manifest_sha256(trajectory_ids),
        "encoder_checkpoint": str(checkpoint),
    }
    cache_path = tmp_path / "cache.h5"
    writer = LatentCacheWriter(
        cache_path,
        latent_dim=2,
        config_yaml=yaml.safe_dump(cache_config),
        encoder_checkpoint_path=str(checkpoint),
        protocol_metadata=protocol,
    )
    for trajectory_id in trajectory_ids:
        writer.write_trajectory(trajectory_id, jnp.ones((2, 2), dtype=jnp.float32))

    assert pipeline._validate_latent_cache_protocol(
        cache_path,
        expected_seed=52,
        expected_backend="cyclone_kvikio",
        require_complete=True,
    ) == trajectory_ids

    def replace_metadata(*, config_payload=cache_config, protocol_payload=protocol):
        with h5py.File(cache_path, "a") as handle:
            handle["metadata"].attrs["config_yaml"] = yaml.safe_dump(config_payload)
            handle["metadata"].attrs["protocol_json"] = json.dumps(protocol_payload)

    replace_metadata(config_payload={"training": {"seed": 52}})
    with pytest.raises(ValueError, match="missing its data mapping"):
        pipeline._validate_latent_cache_protocol(
            cache_path, expected_seed=52, expected_backend="cyclone_kvikio", require_complete=True
        )
    replace_metadata(config_payload={**cache_config, "data": {**cache_config["data"], "split": "train"}})
    with pytest.raises(ValueError, match="data.split='all'"):
        pipeline._validate_latent_cache_protocol(
            cache_path, expected_seed=52, expected_backend="cyclone_kvikio", require_complete=True
        )
    replace_metadata(protocol_payload={key: value for key, value in protocol.items() if key != "artifact_role"})
    with pytest.raises(ValueError, match="missing fields"):
        pipeline._validate_latent_cache_protocol(
            cache_path, expected_seed=52, expected_backend="cyclone_kvikio", require_complete=True
        )
    replace_metadata(protocol_payload={**protocol, "universe_trajectory_ids": list(reversed(trajectory_ids))})
    with pytest.raises(ValueError, match="trajectory IDs differ"):
        pipeline._validate_latent_cache_protocol(
            cache_path, expected_seed=52, expected_backend="cyclone_kvikio", require_complete=True
        )
    replace_metadata(protocol_payload={**protocol, "universe_manifest_sha256": "tampered"})
    with pytest.raises(ValueError, match="trajectory manifest"):
        pipeline._validate_latent_cache_protocol(
            cache_path, expected_seed=52, expected_backend="cyclone_kvikio", require_complete=True
        )
    replace_metadata(protocol_payload={**protocol, "encoder_checkpoint": "different"})
    with pytest.raises(ValueError, match="consistent encoder checkpoint"):
        pipeline._validate_latent_cache_protocol(
            cache_path, expected_seed=52, expected_backend="cyclone_kvikio", require_complete=True
        )

    legacy_path = tmp_path / "legacy.h5"
    legacy = LatentCacheWriter(legacy_path, latent_dim=2, config_yaml=yaml.safe_dump(cache_config))
    legacy.write_trajectory("traj-a", jnp.ones((2, 2), dtype=jnp.float32))
    with pytest.raises(ValueError, match="missing resolved config or protocol metadata"):
        pipeline._validate_latent_cache_protocol(
            legacy_path,
            expected_seed=52,
            expected_backend="cyclone_kvikio",
            require_complete=True,
        )
    assert pipeline._validate_latent_cache_protocol(
        legacy_path,
        expected_seed=52,
        expected_backend="synthetic",
        require_complete=False,
    ) == ("traj-a",)
    assert pipeline._metadata_data_seed(None) is None
    assert pipeline._metadata_data_seed({}) is None
    assert pipeline._protocol_tuple({"ids": [1]}, "ids") is None
    empty_metadata = tmp_path / "empty_metadata.h5"
    LatentCacheWriter(empty_metadata, latent_dim=2)
    assert pipeline._latent_cache_run_config(empty_metadata) is None
    assert pipeline._latent_cache_protocol(empty_metadata) is None


def test_config_loader_and_validation_remaining_command_edges(tiny_config_path):
    with pytest.raises(ValueError, match="override must use key=value"):
        load_config(tiny_config_path, overrides=["training.max_steps"], command="train-encoder")
    with pytest.raises(ValueError, match="non-mapping"):
        load_config(tiny_config_path, overrides=["data.backend.kind=h5"], command="train-encoder")
    with pytest.raises(ValueError, match="simsiam loss"):
        load_config(tiny_config_path, overrides=["loss.simsiam_weight=1.0"])
    with pytest.raises(ValueError, match="latent_cache.path"):
        load_config(
            tiny_config_path,
            overrides=[
                "model.sequence={type: mlp_delta, latent_dim: 32, context_length: 4, hidden_dims: [16], extra: {}}"
            ],
            command="train-sequence",
        )
    with pytest.raises(ValueError, match="evaluate-rollout requires latent_cache.path"):
        load_config(tiny_config_path, command="evaluate-rollout")
    with pytest.raises(ValueError, match="sequence_checkpoint_path"):
        load_config(
            tiny_config_path,
            overrides=["latent_cache.path=/tmp/nonexistent_latent_cache.h5"],
            command="evaluate-rollout",
        )
    config = load_config(
        tiny_config_path,
        overrides=[
            "latent_cache.path=/tmp/nonexistent_latent_cache.h5",
            "latent_cache.use_persistence_baseline=true",
        ],
        command="evaluate-rollout",
    )
    assert config.latent_cache.use_persistence_baseline


def test_train_sequence_private_branches_are_shape_safe():
    context = _latent_context()
    target = jnp.ones((2, 1, 4), dtype=jnp.float32)

    assert sequence_train._batch_get({"inputs": context}, "context", "inputs") is context
    with pytest.raises(KeyError, match="batch is missing"):
        sequence_train._batch_get({}, "context")
    assert sequence_train._cfg({"loss": {"latent_loss": "huber"}}, "latent_loss", "mse") == "huber"
    assert sequence_train._cfg({"latent_loss": "cosine"}, "latent_loss", "mse") == "cosine"

    def with_prediction_length(variables, value, *, train, prediction_length):
        del variables, train
        return {"prediction": jnp.repeat(value[:, -1:, :], prediction_length, axis=1)}

    def without_prediction_length(variables, value, *, train):
        del variables, train
        return {"pred": value[:, -1, :]}

    def without_train(variables, value):
        del variables
        return {"latent": value[:, -1, :]}

    def raw_params_only(params, value):
        if isinstance(params, dict) and "params" in params:
            raise TypeError("raw params only")
        return value[:, -1, :]

    assert sequence_train._prediction_array(
        sequence_train._call_apply(with_prediction_length, {}, context, train=True, prediction_length=2)
    ).shape == (2, 2, 4)
    assert sequence_train._prediction_array(
        sequence_train._call_apply(without_prediction_length, {}, context, train=True, prediction_length=2)
    ).shape == (2, 4)
    assert sequence_train._prediction_array(
        sequence_train._call_apply(without_train, {}, context, train=True, prediction_length=2)
    ).shape == (2, 4)
    assert sequence_train._call_apply(raw_params_only, {"w": 1}, context, train=True, prediction_length=2).shape == (
        2,
        4,
    )
    with pytest.raises(KeyError, match="no prediction key"):
        sequence_train._prediction_array({"bad": context})

    assert jnp.isfinite(sequence_train.latent_prediction_loss(context[:, -1, :], target, mode="mse"))
    assert jnp.isfinite(sequence_train.latent_prediction_loss(target, context[:, -1, :], mode="huber"))
    assert jnp.isfinite(sequence_train.latent_prediction_loss(target, context[:, -1, :], mode="cosine"))
    assert jnp.isfinite(sequence_train.latent_prediction_loss(target, context[:, -1, :], mode="mse_plus_cosine"))
    with pytest.raises(ValueError, match="Unsupported latent loss"):
        sequence_train.latent_prediction_loss(target, context[:, -1, :], mode="bad")
    assert sequence_train.persistence_predict(context, prediction_length=2).shape == (2, 2, 4)


def test_train_encoder_helper_branches_are_explicit():
    z = jnp.ones((2, 4), dtype=jnp.float32)
    diagnostics = SimpleNamespace(flux=z[:, :1], spectra={"ky": z})
    output = SimpleNamespace(z=z, diagnostics=diagnostics, projection=z + 1.0, prediction=z + 2.0)
    mapped = encoder_train._as_mapping(output)
    assert mapped["flux"].shape == (2, 1)
    assert mapped["projection"].shape == (2, 4)
    assert encoder_train._as_mapping((z, {"flux": z}))["z"].shape == (2, 4)
    assert encoder_train._as_mapping((z, z + 1.0, z + 2.0))["prediction"].shape == (2, 4)
    assert encoder_train._as_mapping(z)["z"].shape == (2, 4)
    assert encoder_train._maybe_batch_get({}, "missing") is None
    with pytest.raises(KeyError, match="batch is missing"):
        encoder_train._batch_get({}, "x")

    assert encoder_train._spectra_loss({}, {}, log_space=False, eps=1e-6) == 0.0
    with pytest.raises(KeyError, match="missing key"):
        encoder_train._spectra_loss({}, {"ky": z}, log_space=False, eps=1e-6)
    assert jnp.isfinite(encoder_train._spectra_loss({"ky": -z}, {"ky": z}, log_space=True, eps=1e-6))
    view1, view2 = encoder_train._augment_pair(z, jax.random.PRNGKey(0), 0.0)
    assert jnp.allclose(view1, z)
    assert jnp.allclose(view2, z)
    noisy1, noisy2 = encoder_train._augment_pair(
        z.reshape((2, 1, 2, 2)),
        jax.random.PRNGKey(0),
        {
            "augmentations": {
                "gaussian_noise_std": 0.1,
                "amplitude_jitter_std": 0.1,
                "mask_probability": 0.1,
                "channel_dropout_probability": 0.1,
            }
        },
    )
    assert noisy1.shape == noisy2.shape == (2, 1, 2, 2)

    def with_dropout(variables, value, *, train, rngs):
        del variables, train, rngs
        return value

    def without_rng(variables, value, *, train):
        del variables, train
        return value

    def without_train(variables, value):
        del variables
        return value

    def raw_params_only(params, value):
        if isinstance(params, dict) and "params" in params:
            raise TypeError("raw params only")
        return value

    assert encoder_train._call_apply(with_dropout, {}, z, train=True, rng=jax.random.PRNGKey(1)).shape == z.shape
    assert encoder_train._call_apply(without_rng, {}, z, train=True, rng=jax.random.PRNGKey(1)).shape == z.shape
    assert encoder_train._call_apply(without_train, {}, z, train=True).shape == z.shape
    assert encoder_train._call_apply(raw_params_only, {"w": 1}, z, train=True).shape == z.shape


def test_evaluation_rollout_error_and_fallback_branches():
    context = _latent_context()

    def with_train(variables, value, *, train=False):
        del variables, train
        return {"prediction": value[:, -1, :] + 1.0}

    def without_train(variables, value):
        del variables
        return {"z": value[:, -1, :]}

    def raw_params_only(params, value):
        if isinstance(params, dict) and "params" in params:
            raise TypeError("raw params only")
        return {"latent": value[:, -1, :]}

    assert eval_rollout._prediction_array(eval_rollout._call_model(with_train, {}, context, train=False)).shape == (
        2,
        4,
    )
    assert eval_rollout._prediction_array(eval_rollout._call_model(without_train, {}, context, train=False)).shape == (
        2,
        4,
    )
    assert eval_rollout._prediction_array(
        eval_rollout._call_model(raw_params_only, {"w": 1}, context, train=False)
    ).shape == (2, 4)
    with pytest.raises(KeyError, match="no prediction key"):
        eval_rollout._prediction_array({"bad": context})
    with pytest.raises(ValueError, match="positive"):
        eval_rollout.autoregressive_rollout(with_train, {}, context, rollout_steps=0)
    with pytest.raises(KeyError, match="context/target"):
        eval_rollout.evaluate_rollout_batches(with_train, {}, [{"context": context}], rollout_steps=1)
    with pytest.raises(ValueError, match="no rollout batches"):
        eval_rollout.evaluate_rollout_batches(with_train, {}, [], rollout_steps=1)
    metrics = eval_rollout.evaluate_rollout_batches(
        with_train,
        {},
        [{"context": context, "target": jnp.ones((2, 2, 4), dtype=jnp.float32)}],
        rollout_steps=2,
    )
    assert metrics["mse_by_step"].shape == (2,)
    assert eval_rollout.horizon_until_threshold(jnp.asarray([0.1, 0.2]), threshold=1.0) == 2


def test_encoder_validation_and_dropout_branches():
    x = _snapshot_batch()
    for name in ("relu", "swish", "silu", "tanh", "linear", "identity", "none"):
        assert _activation(name)(jnp.asarray([1.0])).shape == (1,)
    with pytest.raises(ValueError, match="Unsupported activation"):
        _activation("bad")
    assert _as_tuple(2, length=5, name="patch") == (2, 2, 2, 2, 2)
    with pytest.raises(ValueError, match="must have length"):
        _as_tuple((1, 2), length=5, name="patch")

    with pytest.raises(ValueError, match="shape"):
        FlattenMLPEncoder(latent_dim=4).init(jax.random.PRNGKey(0), jnp.ones((2, 2)), train=False)
    with pytest.raises(ValueError, match="latent_dim"):
        FlattenMLPEncoder(latent_dim=0).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="latent_dim"):
        ConvNDEncoder(latent_dim=0).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="channels"):
        ConvNDEncoder(latent_dim=4, channels=()).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="kernel_size"):
        ConvNDEncoder(latent_dim=4, channels=(4,), kernel_size=(1, 2)).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="one entry"):
        ConvNDEncoder(
            latent_dim=4,
            channels=(4,),
            strides=((1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
        ).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="length 5"):
        ConvNDEncoder(latent_dim=4, channels=(4,), strides=((1, 1),)).init(jax.random.PRNGKey(0), x, train=False)

    with pytest.raises(ValueError, match="latent_dim"):
        PatchTransformerEncoder(latent_dim=0).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="patch_size"):
        PatchTransformerEncoder(latent_dim=4, patch_size=(2, 2)).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="num_heads"):
        PatchTransformerEncoder(latent_dim=4, embed_dim=10, num_heads=3).init(jax.random.PRNGKey(0), x, train=False)
    with pytest.raises(ValueError, match="exceeds"):
        PatchTransformerEncoder(latent_dim=4, embed_dim=8, num_heads=2, max_token_count=1).init(
            jax.random.PRNGKey(0), x, train=False
        )
    model = PatchTransformerEncoder(
        latent_dim=4,
        patch_size=(2, 2, 2, 2, 2),
        embed_dim=8,
        depth=1,
        num_heads=2,
        dropout_rate=0.1,
        use_cls_token=True,
        max_token_count=1,
        allow_large_token_count=True,
    )
    variables = model.init(jax.random.PRNGKey(1), x, train=True)
    out = model.apply(variables, x, train=True, rngs={"dropout": jax.random.PRNGKey(2)})
    assert out.shape == (2, 4)
