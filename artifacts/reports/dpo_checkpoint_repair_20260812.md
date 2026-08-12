# DPO full-run checkpoint repair (2026-08-12)

## Failure classification

- Training was finite and made normal progress through optimizer step 283.
- The failure occurred before checkpoint tensor files were written, in
  `torch.distributed.checkpoint` planner `scatter_object_list`, using the
  default NCCL process group.
- Storage was healthy, GPUs 1/2 had no Xid/remap errors, GPUs 1/2 were linked
  by NVLink, and only two GPUs were used. The failure is therefore classified
  as checkpoint-control-plane infrastructure, not a DPO/data/resource-policy
  failure.

## Repair

- FSDP training collectives remain on NCCL.
- DCP planner/object collectives use a dedicated CPU/Gloo process group for
  both save and load.
- CUDA work is synchronized and unused cached allocator blocks are released
  before checkpoint construction.
- Only sealed, checksum-validated checkpoints are eligible for resume.
- Launcher and trainer persist failure status instead of leaving stale
  `running` manifests.
- Full-run launcher accepts an explicit variant subset, enabling a DPO-only
  gate before downstream baselines.

No baseline formula, dataset row/order, reference cache, seed, optimizer,
learning rate, effective batch, scheduler, or training budget changed.

## Verification

- Server unit suite: 27 passed.
- Two-GPU FSDP gate on GPUs 1/2:
  - step 1 trained and wrote a ~92 GB model/optimizer/scheduler/RNG checkpoint;
  - checkpoint received a trusted-local seal;
  - a new process restored checkpoint-1 and trained/saved through step 2;
  - resolved config recorded `checkpoint_metadata_backend=gloo`;
  - no NCCL/CUDA error occurred.

The repair is verified for save/resume. A DPO-only seed-42 full rerun remains
the final long-duration gate before MMPO/ODPO/Scaled DPO are released.
