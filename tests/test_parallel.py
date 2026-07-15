from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.config.schema import ParallelConfig
from gk_surrogate.data.latent_cache import LatentCacheWriter
from gk_surrogate.parallel.batch import (
    drop_or_pad_to_multiple,
    infer_global_and_per_device_batch_size,
    shard_batch,
    unshard_batch,
)
from gk_surrogate.parallel.devices import resolve_parallel_mode, write_device_report
from gk_surrogate.parallel.pmap_steps import (
    pmap_encoder_eval_step,
    pmap_encoder_train_step,
    pmap_sequence_eval_step,
    pmap_sequence_train_step,
)
from gk_surrogate.parallel.replicate import (
    replicate_params,
    replicate_state,
    unreplicate_params,
    unreplicate_state,
    unreplicate_tree,
)
from gk_surrogate.pipeline import _init_encoder_state, _init_sequence_state, _sequence_batches, _snapshot_batches
from gk_surrogate.utils.pretty import scalarize


def test_parallel_config_defaults_and_override(tiny_config_path):
    default_config = load_config(tiny_config_path, command="train-encoder")
    assert default_config.parallel.mode == "auto"
    config = load_config(
        tiny_config_path,
        overrides=["parallel.mode=auto", "parallel.drop_remainder=false"],
        command="train-encoder",
    )
    assert config.parallel.mode == "auto"
    assert config.parallel.drop_remainder is False
    assert config.parallel.axis_name == "devices"
    with pytest.raises(ValueError, match="axis_name"):
        ParallelConfig(axis_name="")


def test_batch_sharding_drop_and_pad_roundtrip():
    batch = {
        "x": jnp.arange(5 * 2, dtype=jnp.float32).reshape(5, 2),
        "spectra": {"ky": jnp.ones((5, 3), dtype=jnp.float32)},
        "flux": None,
    }
    dropped = drop_or_pad_to_multiple(batch, 2, drop_remainder=True)
    assert dropped is not None
    assert dropped["x"].shape == (4, 2)
    assert drop_or_pad_to_multiple({"x": jnp.ones((1, 2))}, 2, drop_remainder=True) is None
    padded = drop_or_pad_to_multiple(batch, 2, drop_remainder=False)
    assert padded is not None
    assert padded["x"].shape == (6, 2)
    assert jnp.allclose(padded["x"][-1], batch["x"][-1])
    assert infer_global_and_per_device_batch_size(dropped, 2) == (4, 2)
    sharded = shard_batch(dropped, 2)
    assert sharded["x"].shape == (2, 2, 2)
    assert unshard_batch(sharded)["x"].shape == (4, 2)
    with pytest.raises(ValueError, match="positive"):
        shard_batch(batch, 0)
    with pytest.raises(ValueError, match="divisible"):
        infer_global_and_per_device_batch_size(batch, 2)
    with pytest.raises(ValueError, match="leading batch"):
        shard_batch({"x": jnp.asarray(1.0)}, 1)
    with pytest.raises(ValueError, match="device and per-device"):
        unshard_batch({"x": jnp.ones((2,), dtype=jnp.float32)})
    with pytest.raises(ValueError, match="multiple"):
        drop_or_pad_to_multiple(batch, 0)
    with pytest.raises(ValueError, match="no array leaves"):
        drop_or_pad_to_multiple({"x": None}, 2)
    with pytest.raises(ValueError, match="positive"):
        drop_or_pad_to_multiple({"x": jnp.ones((0, 2), dtype=jnp.float32)}, 2, drop_remainder=False)


def test_resolve_parallel_modes_and_device_report(tmp_path, monkeypatch):
    import gk_surrogate.parallel.devices as device_helpers

    monkeypatch.setattr("gk_surrogate.parallel.devices.get_local_devices", lambda: ("d0", "d1", "d2", "d3"))
    assert device_helpers.get_device_count() == 4
    single = resolve_parallel_mode(ParallelConfig(mode="single"), batch_size=8)
    assert single.mode == "single"
    auto = resolve_parallel_mode(ParallelConfig(mode="auto"), batch_size=8)
    assert auto.mode == "pmap"
    assert auto.num_devices == 4
    reduced = resolve_parallel_mode(ParallelConfig(mode="auto"), batch_size=6)
    assert reduced.num_devices == 3
    padded = resolve_parallel_mode(ParallelConfig(mode="auto", drop_remainder=False), batch_size=5)
    assert padded.per_device_batch_size == 2
    with pytest.raises(ValueError, match="batch_size"):
        resolve_parallel_mode(ParallelConfig(mode="auto"), batch_size=0)
    with pytest.raises(ValueError, match="not divisible"):
        resolve_parallel_mode(
            ParallelConfig(mode="auto", require_all_visible_devices=True),
            batch_size=6,
        )
    with pytest.raises(ValueError, match="requires at least"):
        resolve_parallel_mode(ParallelConfig(mode="pmap", min_devices=5), batch_size=4)
    report = write_device_report(tmp_path, config=ParallelConfig(mode="single"), plan=single)
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    assert payload["parallel_plan"]["mode"] == "single"


def test_replicate_params_and_scalar_string_summary():
    params = {"w": jnp.arange(3, dtype=jnp.float32)}
    replicated = replicate_params(params, tuple(jax.local_devices()[:1]))
    assert replicated["w"].shape == (1, 3)
    restored = unreplicate_params(replicated)
    assert jnp.allclose(restored["w"], params["w"])
    assert float(unreplicate_tree(jnp.asarray(1.0))) == 1.0
    assert scalarize(["cpu:0", "cpu:1"]) == "cpu:0,cpu:1"


def test_pmap_encoder_step_runs_on_available_cpu_device(tiny_config_path, params_changed):
    config = load_config(tiny_config_path, command="train-encoder")
    state, _ = _init_encoder_state(config)
    batch = next(_snapshot_batches(config, repeat=False))
    replicated = replicate_state(state, tuple(jax.local_devices()[:1]))
    sharded = shard_batch(batch, 1)
    new_state, metrics = pmap_encoder_train_step(replicated, sharded)
    host_state = unreplicate_state(new_state)
    host_metrics = unreplicate_tree(metrics)
    assert int(host_state.step) == 1
    assert params_changed(state.params, host_state.params)
    assert jnp.isfinite(host_metrics["loss"])
    eval_metrics = unreplicate_tree(pmap_encoder_eval_step(new_state, sharded))
    assert jnp.isfinite(eval_metrics["loss"])


def test_pmap_sequence_step_runs_on_available_cpu_device(repo_root, tmp_path, params_changed):
    config = load_config(repo_root / "configs/experiment/smoke_sequence.yaml", command="train-sequence")
    cache_path = tmp_path / "latent_cache.h5"
    writer = LatentCacheWriter(cache_path, latent_dim=32)
    writer.write_trajectory("traj", np.ones((12, 32), dtype=np.float32))
    config = config.model_copy(
        update={"latent_cache": config.latent_cache.model_copy(update={"path": str(cache_path)})}
    )
    state, _ = _init_sequence_state(config)
    batch = next(_sequence_batches(config, cache_path, repeat=False))
    replicated = replicate_state(state, tuple(jax.local_devices()[:1]))
    sharded = shard_batch(batch, 1)
    new_state, metrics = pmap_sequence_train_step(replicated, sharded)
    host_state = unreplicate_state(new_state)
    host_metrics = unreplicate_tree(metrics)
    assert int(host_state.step) == 1
    assert params_changed(state.params, host_state.params)
    assert jnp.isfinite(host_metrics["loss"])
    eval_metrics = unreplicate_tree(pmap_sequence_eval_step(new_state, sharded))
    assert jnp.isfinite(eval_metrics["loss"])
