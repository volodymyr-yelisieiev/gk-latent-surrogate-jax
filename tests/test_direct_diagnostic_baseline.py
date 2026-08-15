from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from gk_surrogate.config.load import load_config
from gk_surrogate.config.schema import DiagnosticHeadConfig, ParallelConfig
from gk_surrogate.evaluation.direct_diagnostics import evaluate_direct_snapshot_diagnostics
from gk_surrogate.factory import build_direct_diagnostic_baseline
from gk_surrogate.models.diagnostics import DiagnosticHeads, DirectSnapshotDiagnosticBaseline
from gk_surrogate.parallel.devices import resolve_parallel_mode
from gk_surrogate.pipeline import _aggregate_eval_metrics_by_trajectory, train_direct_diagnostics


def test_direct_snapshot_baseline_shapes_jit_and_metrics() -> None:
    x = jnp.ones((3, 2, 2, 2, 2, 2, 2), dtype=jnp.float32)
    model = DirectSnapshotDiagnosticBaseline(
        flux_dim=1,
        spectra_dims={"ky": 5, "q": 7},
        hidden_dims=(16,),
    )
    variables = model.init(jax.random.PRNGKey(0), x, train=False)
    predictions = jax.jit(lambda value: model.apply(variables, value, train=False))(x)

    assert predictions.flux is not None
    assert predictions.flux.shape == (3, 1)
    assert predictions.spectra["ky"].shape == (3, 5)
    assert predictions.spectra["q"].shape == (3, 7)
    assert variables["params"]["direct_diagnostic_heads"]["trunk_dense_0"]["kernel"].shape == (6, 16)

    metrics = evaluate_direct_snapshot_diagnostics(
        model.apply,
        variables,
        x,
        flux_target=jnp.zeros((3, 1), dtype=jnp.float32),
        spectra_target={
            "ky": jnp.ones((3, 5), dtype=jnp.float32),
            "q": jnp.ones((3, 7), dtype=jnp.float32),
        },
    )
    assert jnp.isfinite(metrics["flux/rmse"])
    assert all(jnp.isfinite(value) for value in metrics["spectra/mse"].values())


def test_direct_snapshot_baseline_requires_channel_first_5d_snapshot() -> None:
    model = DirectSnapshotDiagnosticBaseline(flux_dim=1)
    with pytest.raises(ValueError, match=r"\[B, C, S1, S2, S3, S4, S5\]"):
        model.init(jax.random.PRNGKey(1), jnp.ones((2, 3, 4)), train=False)


def test_direct_snapshot_baseline_factory_and_missing_outputs() -> None:
    model = build_direct_diagnostic_baseline(
        DiagnosticHeadConfig(flux_dim=None, spectra_dims={"ky": 4}, hidden_dims=(8,))
    )
    x = jnp.ones((2, 1, 1, 1, 1, 1, 1), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(2), x, train=False)
    with pytest.raises(ValueError, match="no flux output"):
        evaluate_direct_snapshot_diagnostics(
            model.apply,
            variables,
            x,
            flux_target=jnp.ones((2, 1), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="missing spectra outputs: q"):
        evaluate_direct_snapshot_diagnostics(
            model.apply,
            variables,
            x,
            spectra_target={"q": jnp.ones((2, 4), dtype=jnp.float32)},
        )


@pytest.mark.parametrize(
    ("model", "message"),
    (
        (DirectSnapshotDiagnosticBaseline(flux_dim=1, hidden_dims=(0,)), "hidden_dims"),
        (DirectSnapshotDiagnosticBaseline(spectra_dims={"ky": 0}), "must be positive"),
        (
            DirectSnapshotDiagnosticBaseline(spectra_dims={"a-b": 1, "a_b": 1}),
            "must remain unique",
        ),
    ),
)
def test_direct_snapshot_baseline_rejects_invalid_head_contracts(
    model: DirectSnapshotDiagnosticBaseline,
    message: str,
) -> None:
    snapshot = jnp.ones((2, 1, 2, 2, 2, 2, 2))
    with pytest.raises(ValueError, match=message):
        model.init(jax.random.PRNGKey(3), snapshot, train=False)


def test_diagnostic_heads_reject_non_latent_input_and_apply_dropout() -> None:
    heads = DiagnosticHeads(flux_dim=1, hidden_dims=(4,), dropout_rate=0.25)
    with pytest.raises(ValueError, match=r"\[B, Z\]"):
        heads.init(jax.random.PRNGKey(4), jnp.ones((2, 3, 4)), train=False)

    z = jnp.ones((2, 3))
    variables = heads.init({"params": jax.random.PRNGKey(5), "dropout": jax.random.PRNGKey(6)}, z, train=True)
    predictions = heads.apply(
        variables,
        z,
        train=True,
        rngs={"dropout": jax.random.PRNGKey(7)},
    )
    assert predictions.flux is not None
    assert predictions.flux.shape == (2, 1)


def test_direct_diagnostic_pipeline_trains_and_reports_held_out_validation(repo_root, tmp_path) -> None:
    config = load_config(
        repo_root / "configs/experiment/smoke_encoder_supervised.yaml",
        overrides=["training.max_steps=2", "training.eval_every=1", "training.checkpoint_every=2"],
        command="train-direct-diagnostics",
    )
    config = config.model_copy(update={"output_dir": str(tmp_path / "direct")})
    result = train_direct_diagnostics(config)

    assert result["artifact_role"] == "direct_diagnostic_checkpoint"
    assert result["test_split_inspected"] is False
    assert set(result["train_trajectory_ids"]).isdisjoint(result["validation_trajectory_ids"])
    assert result["checkpoint_selection"] == "validation_trajectory_balanced_flux_rmse"
    assert result["validation"]["flux_rmse"] >= 0.0
    assert (tmp_path / "direct" / "metrics.json").is_file()
    assert (tmp_path / "direct" / "normalization_stats.npz").is_file()


def test_validation_aggregation_gives_each_trajectory_equal_weight() -> None:
    batches = {
        "short": ({"x": jnp.zeros((1, 1), dtype=jnp.float32)},),
        "long": ({"x": jnp.full((3, 1), 10.0, dtype=jnp.float32)},),
    }
    plan = resolve_parallel_mode(ParallelConfig(mode="single"), batch_size=3)

    metrics = _aggregate_eval_metrics_by_trajectory(
        None,
        ("short", "long"),
        batches_for_trajectory=lambda trajectory_id: iter(batches[trajectory_id]),
        plan=plan,
        step_fn=lambda _state, batch: {"value": jnp.mean(batch["x"])},
    )

    assert metrics["value"] == pytest.approx(5.0)
