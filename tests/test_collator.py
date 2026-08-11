import torch

from rubric_dpo.collator import BaselinePreferenceCollator


def test_collator_preserves_baseline_sidecars():
    examples = [
        {"pair_id": "a", "prompt_input_ids": [1, 2], "chosen_input_ids": [3], "rejected_input_ids": [4, 5],
         "margin_raw": 0.5, "margin_normalized": 0.25, "sample_weight": 0.75,
         "ref_chosen_logps": -2.0, "ref_rejected_logps": -3.0},
        {"pair_id": "b", "prompt_input_ids": [6], "chosen_input_ids": [7, 8], "rejected_input_ids": [9],
         "margin_raw": 1.0, "margin_normalized": 0.5, "sample_weight": 1.25,
         "ref_chosen_logps": -4.0, "ref_rejected_logps": -5.0},
    ]
    batch = BaselinePreferenceCollator(pad_token_id=0)(examples)
    assert batch["pair_id"] == ["a", "b"]
    torch.testing.assert_close(batch["margin_normalized"], torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(batch["sample_weight"], torch.tensor([0.75, 1.25]))
    assert batch["prompt_input_ids"].shape == (2, 2)
