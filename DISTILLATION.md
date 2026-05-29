# EUPE Multi-Stage Distillation

Scaffold for reproducing EUPE's scaling-up → scaling-down distillation. See the design spec
(`docs/superpowers/specs/2026-05-29-multistage-multiteacher-distillation-design.md`) and the
engineering report (`../EUPE_Distillation_Reproduction_Report.md`) for full rationale.

> **Status: scaffold.** All Python bodies raise `NotImplementedError`; fill them in following the
> docstring references before running. Configs are concrete (paths are `<PLACEHOLDERS>`).

## Pipeline
1. **Stage 1** — distill PEcore-G + PElang-G + DINOv3-H+ into a ~1.9B ViT-G proxy.
2. **Stage 2** — distill the frozen proxy into the efficient family @256, bs 8192, lr 2e-5, 390k iters.
3. **Stage 3** — multi-resolution finetune @{256,384,512}, bs 4096, lr 1e-5, 100k iters (init from Stage 2).

## Launch
```bash
# Stage 1 (single student = proxy)
python -m eupe.run.submit eupe/train/train.py --nodes 32 --ngpus 8 \
  --config-file eupe/configs/train/stage1_multiteacher_proxy.yaml --output-dir <OUT>

# Stage 2 / 3 (co-distill the family; --multi-distillation enables rank-subgroups)
python -m eupe.run.submit eupe/train/train.py --nodes 16 --ngpus 8 --multi-distillation \
  --config-file eupe/configs/train/stage2_multidistill.yaml --output-dir <OUT>
```

## Fill-in checklist (each maps to one `NotImplementedError`)
- `eupe/distill/adapters.py` — 2-layer MLP (paper §4.1)
- `eupe/distill/normalize.py` — frozen mean/std + 500-iter estimator (paper §3.3)
- `eupe/distill/loss.py` — cosine + 0.9·cos/0.1·smoothL1 (paper §3.2, Eq. 4-7)
- `eupe/distill/teachers.py` — load PEcore-G/PElang-G (perception_models), DINOv3-H+ (dinov3), proxy
- `eupe/train/{param_groups,cosine_lr_scheduler}.py` — optimizer groups + schedules (dinov3)
- `eupe/train/{distill,multidist}_meta_arch.py` — step orchestration + subgroup broadcast (dinov3 multidist_meta_arch.py)
- `eupe/train/train.py` — loop + checkpointing
- `eupe/fsdp/ac_compile_parallelize.py` — FSDP(SHARD_GRAD_OP) + ac + compile (dinov3 fsdp)
- `eupe/data/distillation_loaders.py` — LVD+IN1k mix (p=0.10) + pyramid collate

## Validation milestones
- After Stage 1: reproduce proxy numbers (report Table 4).
- After Stage 2: "Stage 1&2" column (report Table 2).
- After Stage 3: final EUPE numbers (report Table 1), then evaluate with `eupe/eval/*`.

## LR scaling note
`eupe/configs/config.py` applies `sqrt_wrt_1024` to `optim.lr`. The stage configs use schedules-v2
(`schedules.lr.peak`) to set the peak LR directly (2e-5 / 1e-5) regardless of GPU count.
