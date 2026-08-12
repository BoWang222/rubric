from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch.distributed as dist


def create_checkpoint_process_group():
    """Create a CPU process group used only for DCP metadata collectives."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    return dist.new_group(backend="gloo")


@contextmanager
def use_checkpoint_process_group(process_group) -> Iterator[None]:
    """Route torch.distributed.checkpoint coordination away from the CUDA/NCCL group.

    FSDP tensor collectives continue to use the default NCCL group.  Only DCP's
    planner/object collectives use this CPU/Gloo group, avoiding CUDA allocations
    for pickled checkpoint metadata when training already occupies most of HBM.
    """
    if process_group is None:
        yield
        return

    import torch.distributed.checkpoint as dist_cp

    original_save = dist_cp.save
    original_load = dist_cp.load

    def save_with_checkpoint_group(*args, **kwargs):
        kwargs.setdefault("process_group", process_group)
        return original_save(*args, **kwargs)

    def load_with_checkpoint_group(*args, **kwargs):
        kwargs.setdefault("process_group", process_group)
        return original_load(*args, **kwargs)

    dist_cp.save = save_with_checkpoint_group
    dist_cp.load = load_with_checkpoint_group
    try:
        yield
    finally:
        dist_cp.save = original_save
        dist_cp.load = original_load
