# Rubric-aware preference optimization

Controlled Qwen3-8B preference-optimization experiments on UltraFeedback.

Implemented baseline variants:

- Vanilla DPO
- MMPO
- ODPO log-gap adaptation
- Scaled-DPO normalized-gap transfer

The repository contains the audited UltraFeedback preprocessing/tokenization
pipeline, durable reference-logp cache builder, a unified TRL 0.19.1 trainer,
FSDP launch profiles, checkpoint verification/finalization, and source-formula
parity tests. Models, datasets, caches, reference repositories, and run outputs
are intentionally excluded from Git.

Start with:

- [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for the experiment gates and commands.
- [`MIGRATION_NUS.md`](MIGRATION_NUS.md) for the H100 server bootstrap.
- [`PLAN.md`](PLAN.md) and [`CHECKLIST.md`](CHECKLIST.md) for the research contract.
