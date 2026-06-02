# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EUPE (Efficient Universal Perception Encoder) — reference PyTorch implementation of a multi-stage
**feature distillation** recipe (arXiv:2603.22387). Multiple frozen foundation teachers are distilled
into a large proxy, which is then distilled down into an efficient ViT/ConvNeXt family. The training
pipeline (`eupe/distill`, `eupe/train`, `eupe/fsdp`, `eupe/data/distillation_loaders.py`) is the
actively developed surface; `eupe/eval` and `eupe/hub` are mostly ported as-is from DINOv3 for
benchmarking and loading released weights. See `DISTILLATION.md` for the run recipe and
`docs/superpowers/specs/` + `docs/superpowers/plans/` for the design spec and implementation plan.

## Commands

```bash
# Environment (Linux + CUDA expected; PyTorch >= 2.7.1, Python >= 3.11)
micromamba env create -f conda.yaml && micromamba activate eupe   # recommended
pip install -e .                                                  # alternative

# Tests (pytest; tests live under tests/, not in the package)
python -m pytest tests/                       # full suite
python -m pytest tests/distill/test_loss.py   # one file
python -m pytest tests/distill/test_loss.py::test_cosine_loss_bounds   # one test

# Lint / type-check (config in pyproject.toml; line-length 120, py311)
ruff check eupe
mypy            # files = "eupe", set in pyproject
pylint eupe     # only similarities + FIXME/XXX/TODO notes are enabled
```

### Training (launched via the submitit wrapper `eupe.run.submit`)

`python -m eupe.run.submit <script.py> --nodes N --ngpus 8 [submitit opts] -- <script opts>`
submits a SLURM job that runs `<script>.main(script_args)`. The training entry point is
`eupe/train/train.py`:

```bash
# Stage 1 — 3 foundation teachers -> ViT-G proxy (single student)
python -m eupe.run.submit eupe/train/train.py --nodes 32 --ngpus 8 \
  --config-file eupe/configs/train/stage1_multiteacher_proxy.yaml --output-dir <OUT>

# Stage 2/3 — co-distill the whole efficient family from the frozen proxy.
# --multi-distillation partitions ranks into per-student subgroups (see config below).
python -m eupe.run.submit eupe/train/train.py --nodes 16 --ngpus 8 --multi-distillation \
  --config-file eupe/configs/train/stage2_multidistill.yaml --output-dir <OUT>
```

Runs auto-resume from the newest `<OUT>/.../ckpt/training_*.pth`; pass `--no-resume` to force fresh.

### Evaluation (ported from DINOv3; run on a single node)

```bash
PYTHONPATH=. python eupe/eval/segmentation/run.py model.eupe_hub=eupe_vitb16 \
  model.pretrained_weights=<CKPT> config=eupe/eval/segmentation/configs/config-ade20k-linear-training.yaml ...
PYTHONPATH=. python eupe/eval/depth/run.py        model.eupe_hub=eupe_vitb16 model.pretrained_weights=<CKPT> ...
python -m eupe.run.submit eupe/eval/knn.py        model.eupe_hub=eupe_vitb16 model.pretrained_weights=<CKPT> ...
```

## Architecture

### The distillation objective (`eupe/distill/`)

Replaces DINOv3's DINO/iBOT/Sinkhorn SSL objective with RADIO-style feature matching. Four pieces:

- **`teachers.py`** — frozen teacher models behind a uniform `TeacherModel` interface
  (`forward(img) -> {"cls": [B,d], "patch": [B,N,d]}`, plus `native_resolution` / `embed_dim`).
  `build_teachers(cfg)` reads `cfg.distill.teachers` and dispatches via `_TEACHER_REGISTRY`. External
  teachers are **lazily imported** (perception_models for `PE{Core,Lang}Teacher`, `torch.hub` for
  `DINOv3Teacher`) so the package imports without them installed. `ProxyTeacher` reloads a Stage-1
  EUPE checkpoint as the Stage 2/3 teacher.
- **`adapters.py`** — `AdapterHeadSet`: one (cls, patch) MLP pair per teacher projecting the student
  dim → teacher dim. Trainable; lives **outside** the FSDP-wrapped student.
- **`normalize.py`** — `FeatureNormalizer` (buffers only) + `estimate_teacher_statistics`. Per-teacher
  per-token mean/std are estimated over a warmup and then **frozen**; teacher targets are standardized
  before the loss.
- **`loss.py`** — `DistillationLoss`: per teacher, `cosine(cls)` + `alpha*cosine(patch) +
  beta*smooth_l1(patch)` (alpha=0.9, beta=0.1), with an optional gamma multiplier on the DINOv3
  teacher's patch term. Patch grids are bicubically aligned to `max(N_student, N_teacher)`.

### Meta-architectures (`eupe/train/`)

- **`DistillationMetaArch`** (`distill_meta_arch.py`) — N frozen teachers → 1 trainable student. Holds
  student + teachers + adapters + normalizers + loss. `forward_backward` does student forward → adapt
  → normalize teachers → loss → backward. Used directly for Stage 1.
- **`MultiDistillationMetaArch`** (`multidist_meta_arch.py`) — subclass for Stage 2/3: one proxy → M
  students on **GPU rank-subgroups**. The proxy runs once on the global batch on all ranks;
  `broadcast_to_subgroups` all-gathers + re-partitions teacher outputs to each student's subgroup.
  Implements the Stage-3 resolution pyramid (teacher and student each sample a scale per iteration,
  seeded by `iteration` so all ranks agree on the spatial shapes used in the cross-rank all-gather).

### Training loop (`eupe/train/train.py`)

`main()` → choose meta-arch (`MultiDistillationMetaArch` iff `cfg.multidistillation.enabled`) →
build on **meta device** → FSDP2-wrap **only the student** (`eupe/fsdp/parallelize`) → materialize +
`init_weights()` → `do_train`. `do_train` builds layer-wise-decayed AdamW param groups (student) plus
separate adapter groups, runs the normalizer warmup, then iterates the mixed data loader applying
cosine LR/WD schedules. **Build ordering is load-bearing**: teachers load real checkpoints onto CUDA
eagerly while the (possibly ~1.9B) student stays on meta until sharded — never `to_empty()` the whole
model (it would wipe the teachers' loaded weights).

### Checkpoints & resume

`_save_checkpoint` writes `{"teacher": <student sd re-keyed under 'teacher.'>, "optimizer",
"normalizers", "adapters", "iteration"}`. Re-keying under `teacher.` lets `ProxyTeacher` and
`build_model_for_eval` reload the trained student. DTensor consolidation (`full_tensor()`,
`get_optimizer_state_dict`) are **collectives every subgroup rank must enter** — consolidate on all
ranks first, then gate only `torch.save` on the subgroup-main rank, then barrier. Resume restores
student + adapters + normalizers + optimizer moments and fast-forwards the data sampler.
`extract_backbone_state_dict` (`eupe/models/__init__.py`) normalizes any checkpoint layout and strips
FSDP/AC/compile name decorations.

### Config system (`eupe/configs/`)

OmegaConf. Non-multidist runs merge `ssl_default_config.yaml` ← stage config ← CLI `key=value` opts
(`setup_config`). Multidist runs use `setup_multidistillation`, which enables distributed, builds the
per-student rank subgroups from `multidistillation.students[*].ranks_range`, and derives each
student's `batch_size_per_gpu` from `global_batch_size`. **schedules-v2**: when a `schedules:` block is
present, peak LR/WD are read directly (`schedules.lr.peak`), bypassing the implicit `sqrt_wrt_1024` LR
scaling in `apply_scaling_rules_to_cfg` so reproducers hit the paper's exact LR regardless of GPU
count. Stage configs live in `eupe/configs/train/{stage*,students/,proxy/,teachers/}`.

### Models (`eupe/models/`) & FSDP (`eupe/fsdp/`)

`build_model` builds ViT (`vision_transformer.py`) or ConvNeXt students/teachers; it constructs
`DinoVisionTransformer` directly when explicit structural dims (`embed_dim`/`depth`/`num_heads`) are
present, else dispatches to a named factory (`vit_large`, …). `eupe/fsdp/ac_compile_parallelize.py`
uses **FSDP2 `fully_shard`** (DTensor), per-block sharding + prefetch, with activation checkpointing
and `torch.compile`; `cfg.compute_precision.sharding_strategy` maps to `reshard_after_forward`. The
mesh spans the rank's process subgroup (multidist) or the world (Stage 1).

### Data (`eupe/data/`)

`distillation_loaders.py` interleaves homogeneous ImageNet-1k batches (prob `distill.imagenet_prob`,
default 0.10) with heterogeneous LVD-1689M batches via `MixedSampler` (Bernoulli-per-batch routing
over rank-aware infinite samplers). `cfg.train.dataset_path` is `<LVD_source>+<ImageNet:...>`, each a
`make_dataset` string parsed by `loaders.py::_parse_dataset_str` (`LVD1689M`/`LVD` alias the label-free
`ImageFolder` reader). Single-view augmentation only (no multi-crop).

## Conventions

- **Provenance comments.** Code ported from the reference repos carries a `Ported from refs/<repo>/…`
  comment naming the source and, critically, the **divergences** from it (e.g. "DINO/iBOT stripped",
  "FSDP1 → FSDP2"). When porting or modifying ported code, keep this convention — it is how the
  codebase tracks fidelity to DINOv3/RADIO/perception_models and to the EUPE paper. Reference sources
  live at `/Users/puyangwang/EUPE/refs/` (dinov3, RADIO, perception_models).
- **Placeholder paths.** Config values like `<PATH/TO/...>` / `<STAGE1_PROXY_CKPT.pth>` are treated as
  unset (code checks `str(x).startswith("<")`). Fill them before a real run; teacher checkpoints and
  the LVD dataset are external prerequisites (see `DISTILLATION.md`).
- **Paper-fidelity notes.** Places where the paper is silent (e.g. Stage-1 recipe, smooth-L1
  transition, PE-Lang cls pooling) are flagged inline as `FIDELITY NOTE` / documented placeholders —
  preserve and update these rather than silently changing behavior.
