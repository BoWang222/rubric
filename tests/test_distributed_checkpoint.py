from __future__ import annotations

from unittest.mock import Mock, patch

import torch.distributed.checkpoint as dist_cp

from rubric_dpo.distributed_checkpoint import use_checkpoint_process_group


def test_checkpoint_calls_receive_the_cpu_metadata_group() -> None:
    group = object()
    save = Mock(return_value="saved")
    load = Mock(return_value="loaded")
    with patch.object(dist_cp, "save", save), patch.object(dist_cp, "load", load):
        with use_checkpoint_process_group(group):
            assert dist_cp.save({"x": 1}) == "saved"
            assert dist_cp.load({"x": 1}) == "loaded"
    save.assert_called_once_with({"x": 1}, process_group=group)
    load.assert_called_once_with({"x": 1}, process_group=group)
