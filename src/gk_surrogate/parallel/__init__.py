"""Optional single-host data parallel helpers."""

from gk_surrogate.parallel.batch import (
    drop_or_pad_to_multiple,
    infer_global_and_per_device_batch_size,
    shard_batch,
    unshard_batch,
)
from gk_surrogate.parallel.devices import (
    ParallelPlan,
    get_device_count,
    get_local_devices,
    resolve_parallel_mode,
    write_device_report,
)
from gk_surrogate.parallel.replicate import (
    replicate_params,
    replicate_state,
    unreplicate_params,
    unreplicate_state,
    unreplicate_tree,
)

__all__ = [
    "ParallelPlan",
    "drop_or_pad_to_multiple",
    "get_device_count",
    "get_local_devices",
    "infer_global_and_per_device_batch_size",
    "replicate_params",
    "replicate_state",
    "resolve_parallel_mode",
    "shard_batch",
    "unreplicate_params",
    "unreplicate_state",
    "unreplicate_tree",
    "unshard_batch",
    "write_device_report",
]
