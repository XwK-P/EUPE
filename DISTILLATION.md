# EUPE Multi-Stage Distillation

Reproduces EUPE's scaling-up → scaling-down distillation (arXiv:2603.22387v2). See the design spec
(`docs/superpowers/specs/2026-05-29-multistage-multiteacher-distillation-design.md`) and the
engineering report (`../EUPE_Distillation_Reproduction_Report.md`) for full rationale.

> **Status: implemented.** The training pipeline (`eupe/distill`, `eupe/train`, `eupe/fsdp`,
> `eupe/data/distillation_loaders.py`) is complete and unit-tested (`tests/distill`). What remains is
> supplying the external inputs below — datasets, teacher/proxy checkpoints — and tuning the
> paper-silent Stage-1 recipe.

## Pipeline
1. **Stage 1** — distill PEcore-G + PElang-G + DINOv3-H+ into a ~1.9B ViT-G proxy (4 register tokens).
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
Runs resume automatically from the latest `<OUT>/.../ckpt/training_*.pth` (student weights + adapter
heads + optimizer + frozen normalizer stats + iteration); pass `--no-resume` to force a fresh start.

## Prerequisites to fill in before running
- **Teacher checkpoints** — set the `<PATH/TO/...>` fields in `eupe/configs/train/teachers/*.yaml`
  to PEcore-G / PElang-G (facebookresearch/perception_models) and DINOv3-H+ (facebookresearch/dinov3).
- **Datasets** — `train.dataset_path` is `<LVD_source>+<ImageNet:...>` (mixed at `distill.imagenet_prob`,
  default 0.10, paper §3.4). LVD-1689M is proprietary; point the `LVD1689M:root=/PATH` slot at any local
  image tree — it resolves to the label-free `ImageFolder` reader (`eupe/data/datasets/image_folder.py`).
  For LVD-scale runs, add a WebDataset/tar adapter to `eupe/data/loaders.py::_parse_dataset_str`.
- **Stage handoffs** — Stage 2/3 reference the Stage-1 proxy checkpoint (`distill.teachers[0].checkpoint`);
  Stage 3 references each student's Stage-2 checkpoint (`multidistillation.students[*].pretrained_weights`).
- **Stage-1 recipe** — the paper is silent on Stage-1 batch/LR/iterations and exact proxy dims; the values
  in `stage1_multiteacher_proxy.yaml` are documented placeholders. Tune and gate on the Table-4 proxy scores.

## Validation milestones
- After Stage 1: reproduce proxy numbers (report Table 4).
- After Stage 2: "Stage 1&2" column (report Table 2).
- After Stage 3: final EUPE numbers (report Table 1), then evaluate with `eupe/eval/*`.

## LR scaling note
`eupe/configs/config.py` applies `sqrt_wrt_1024` to `optim.lr`. The stage configs use schedules-v2
(`schedules.lr.peak`) to set the peak LR directly (2e-5 / 1e-5) regardless of GPU count.
