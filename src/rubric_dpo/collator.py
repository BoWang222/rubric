from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from trl.trainer.dpo_trainer import DataCollatorForPreference


@dataclass
class BaselinePreferenceCollator(DataCollatorForPreference):
    """TRL preference padding plus immutable baseline sidecar fields."""

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        required = (
            "pair_id", "margin_raw", "margin_normalized", "sample_weight",
            "ref_chosen_logps", "ref_rejected_logps",
        )
        missing = [key for key in required if key not in examples[0]]
        if missing:
            raise KeyError(f"collator input misses required cached fields: {missing}")
        output = super().torch_call(examples)
        output["pair_id"] = [str(example["pair_id"]) for example in examples]
        for key in ("margin_raw", "margin_normalized", "sample_weight"):
            output[key] = torch.tensor([float(example[key]) for example in examples], dtype=torch.float32)
        for key in ("chosen_token_count", "rejected_token_count", "chosen_character_count", "rejected_character_count"):
            if key in examples[0]:
                output[key] = torch.tensor([int(example[key]) for example in examples], dtype=torch.long)
        return output
