from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union

import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from trl import DPOTrainer

from .checkpointing import seal_local_checkpoint, validate_local_checkpoint
from .losses import baseline_losses


class BaselineDPOTrainer(DPOTrainer):
    """One forward path for all four source/formula-aligned baselines."""

    def __init__(
        self,
        *args,
        baseline_variant: str,
        gamma: float = 2.2,
        alpha: float = 0.5,
        target_epsilon: float = 1e-6,
        **kwargs,
    ):
        self.baseline_variant = baseline_variant
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.target_epsilon = float(target_epsilon)
        super().__init__(*args, **kwargs)
        self._precomputed_train_ref_log_probs = True
        self._precomputed_eval_ref_log_probs = True
        if self.ref_model is not None:
            raise AssertionError("durable-cache trainer must not retain a reference model")
        if not self.args.precompute_ref_log_probs:
            raise AssertionError("precompute_ref_log_probs must remain true to suppress reference-model creation")

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        required = {
            "pair_id", "prompt_input_ids", "chosen_input_ids", "rejected_input_ids",
            "margin_raw", "margin_normalized", "sample_weight",
            "ref_chosen_logps", "ref_rejected_logps",
        }
        missing = required - set(dataset.column_names)
        if missing:
            raise ValueError(f"{dataset_name} is not a fully materialized cached dataset; missing {sorted(missing)}")
        if len(set(dataset["pair_id"])) != len(dataset):
            raise ValueError(f"duplicate pair_id in {dataset_name}")
        return dataset

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.is_world_process_zero():
            checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            seal_local_checkpoint(checkpoint, Path(self.args.output_dir))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _load_optimizer_and_scheduler(self, checkpoint):
        import transformers.trainer as transformers_trainer

        if checkpoint is None:
            return super()._load_optimizer_and_scheduler(checkpoint)
        validate_local_checkpoint(Path(checkpoint), Path(self.args.output_dir))
        original_guard = transformers_trainer.check_torch_load_is_safe
        transformers_trainer.check_torch_load_is_safe = lambda: None
        try:
            return super()._load_optimizer_and_scheduler(checkpoint)
        finally:
            transformers_trainer.check_torch_load_is_safe = original_guard

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return super()._load_rng_state(checkpoint)
        validate_local_checkpoint(Path(checkpoint), Path(self.args.output_dir))
        return super()._load_rng_state(checkpoint)

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.Tensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        output = self.concatenated_forward(model, batch)
        result = baseline_losses(
            self.baseline_variant,
            output["chosen_logps"], output["rejected_logps"],
            batch["ref_chosen_logps"], batch["ref_rejected_logps"],
            beta=self.beta,
            margin_normalized=batch["margin_normalized"],
            sample_weight=batch["sample_weight"],
            gamma=self.gamma,
            alpha=self.alpha,
            epsilon=self.target_epsilon,
        )
        reward_accuracies = (result.chosen_rewards > result.rejected_rewards).float()
        prefix = "eval_" if train_eval == "eval" else ""
        gather = self.accelerator.gather_for_metrics
        metrics: dict[str, float] = {
            f"{prefix}rewards/chosen": gather(result.chosen_rewards).mean().item(),
            f"{prefix}rewards/rejected": gather(result.rejected_rewards).mean().item(),
            f"{prefix}rewards/accuracies": gather(reward_accuracies).mean().item(),
            f"{prefix}rewards/margins": gather(result.chosen_rewards - result.rejected_rewards).mean().item(),
            f"{prefix}logps/chosen": gather(output["chosen_logps"]).detach().mean().item(),
            f"{prefix}logps/rejected": gather(output["rejected_logps"]).detach().mean().item(),
            f"{prefix}logits/chosen": gather(output["mean_chosen_logits"]).detach().mean().item(),
            f"{prefix}logits/rejected": gather(output["mean_rejected_logits"]).detach().mean().item(),
            f"{prefix}baseline_loss_unweighted_dpo": gather(torch.nn.functional.softplus(-result.logits)).mean().item(),
            f"{prefix}margin_normalized/mean": gather(batch["margin_normalized"]).mean().item(),
            f"{prefix}policy_ref_logratio/chosen": gather(output["chosen_logps"] - batch["ref_chosen_logps"]).mean().item(),
            f"{prefix}policy_ref_logratio/rejected": gather(output["rejected_logps"] - batch["ref_rejected_logps"]).mean().item(),
        }
        for key in ("chosen_token_count", "rejected_token_count", "chosen_character_count", "rejected_character_count"):
            if key in batch:
                metrics[f"{prefix}response_length/{key}"] = gather(batch[key]).float().mean().item()
        if self.baseline_variant == "mmpo":
            p0 = torch.sigmoid(self.gamma * batch["margin_normalized"]).clamp(
                self.target_epsilon, 1.0 - self.target_epsilon
            )
            gathered_p0 = gather(p0)
            metrics[f"{prefix}mmpo/p0_mean"] = gathered_p0.mean().item()
            metrics[f"{prefix}mmpo/p0_min"] = gathered_p0.min().item()
            metrics[f"{prefix}mmpo/p0_max"] = gathered_p0.max().item()
            metrics[f"{prefix}mmpo/target_clipping_count"] = (
                (gathered_p0 <= self.target_epsilon) | (gathered_p0 >= 1.0 - self.target_epsilon)
            ).sum().float().item()
        elif self.baseline_variant == "odpo_loggap":
            offset = self.alpha * torch.log(batch["margin_normalized"])
            gathered_offset = gather(offset)
            metrics[f"{prefix}odpo/offset_mean"] = gathered_offset.mean().item()
            metrics[f"{prefix}odpo/offset_min"] = gathered_offset.min().item()
            metrics[f"{prefix}odpo/offset_max"] = gathered_offset.max().item()
        elif self.baseline_variant == "scaled_dpo_gap_transfer":
            weights = gather(batch["sample_weight"])
            metrics[f"{prefix}scaled/weight_mean"] = weights.mean().item()
            metrics[f"{prefix}scaled/weight_max"] = weights.max().item()
            ess = weights.sum().square() / weights.square().sum()
            metrics[f"{prefix}scaled/effective_sample_size"] = ess.item()
            metrics[f"{prefix}scaled/ess_fraction"] = (ess / weights.numel()).item()
        return result.losses.mean(), metrics
