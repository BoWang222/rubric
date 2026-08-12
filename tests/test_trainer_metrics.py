import torch

from rubric_dpo.trainer import BaselineDPOTrainer


class _IdentityGatherAccelerator:
    @staticmethod
    def gather_for_metrics(value):
        return value


def test_mmpo_metrics_are_floating_point_for_trl_log_reduction():
    trainer = object.__new__(BaselineDPOTrainer)
    trainer.baseline_variant = "mmpo"
    trainer.gamma = 2.2
    trainer.alpha = 0.5
    trainer.target_epsilon = 1e-6
    trainer.beta = 0.01
    trainer.accelerator = _IdentityGatherAccelerator()
    trainer.concatenated_forward = lambda model, batch: {
        "chosen_logps": torch.tensor([-2.0, -3.0]),
        "rejected_logps": torch.tensor([-2.5, -3.2]),
        "mean_chosen_logits": torch.tensor([-1.0, -1.2]),
        "mean_rejected_logits": torch.tensor([-1.1, -1.3]),
    }
    batch = {
        "ref_chosen_logps": torch.tensor([-2.1, -3.1]),
        "ref_rejected_logps": torch.tensor([-2.6, -3.3]),
        "margin_normalized": torch.tensor([0.25, 1.0]),
        "sample_weight": torch.ones(2),
    }

    _, metrics = trainer.get_batch_loss_metrics(None, batch)

    assert isinstance(metrics["mmpo/target_clipping_count"], float)
    assert all(isinstance(value, float) for value in metrics.values())
