# Design Spec — Multi‑Stage Multi‑Teacher Distillation Scaffold for EUPE

**Date:** 2026‑05‑29
**Status:** Approved design → implementation scaffold
**Repo:** `facebookresearch/EUPE` (branch `feat/multistage-distillation`)
**Recipe reference:** `../EUPE_Distillation_Reproduction_Report.md` (engineering report) and the EUPE paper (arXiv:2603.22387v2). This spec assumes the report for all *rationale*; it specifies the *code structure* only.

---

## 1. Goal & scope

Add a **complete, structurally‑finished scaffold** of the multi‑stage, multi‑teacher distillation pipeline to the EUPE repo, so a team with GPUs + data + teacher checkpoints can fill in the bodies and run it. The official repo ships model definitions + linear eval only; the distillation trainer was stripped out. This scaffold restores it as a DINOv3‑style trainer carrying the RADIO‑style feature‑matching objective.

### Deliverable policy (agreed)
- **Every function/method body is a stub:** signature + a detailed docstring (inputs, outputs, shapes, and the exact paper §/equation or repo file to copy from) + `raise NotImplementedError("TODO: …")`. This includes the loss/adapter/normalizer math.
- **Configs are concrete:** the YAMLs encode the real recipe (batch sizes, LRs, iteration counts, `META_ARCHITECTURE`, rank ranges, resolutions) with `<PATH/…>` placeholders for environment‑specific values.
- **Conventions:** FAIR Research License header on every `.py` (match existing repo files), OmegaConf + dataclass config style, `logging.getLogger("eupe")`, package layout under `eupe/`.
- **Excluded:** no unit tests, no synthetic smoke test, no runnable training, no changes to existing eval code.

### Non‑goals
- Not implementing real training logic, FSDP internals, teacher loading, or data pipelines.
- Not vendoring `perception_models` / `dinov3` — only thin adapter stubs that point at them.
- Not touching `eupe/eval/*` (the shipped linear eval harness stays as‑is).

---

## 2. Architecture overview

```
Stage 1  {PEcore-G, PElang-G, DINOv3-H+}  ──DistillationMetaArch──▶  Proxy (ViT-G ~1.9B)
Stage 2  Proxy (frozen) ──MultiDistillationMetaArch @256, 390k──▶  {ViT-T/S/B, ConvNeXt-T/S/B}
Stage 3  Proxy (frozen) ──MultiDistillationMetaArch @{256,384,512}, 100k──▶  students (init from S2)
```

Two reusable engines cover all three stages:

- **`DistillationMetaArch`** — *N frozen teachers → 1 trainable student.* Used directly for **Stage 1** (N=3 teachers, student = proxy). Also the per‑student unit inside Stage 2/3 (N=1 teacher = proxy).
- **`MultiDistillationMetaArch`** — *1 frozen teacher (proxy) → M students,* each student pinned to a contiguous GPU rank‑span (process subgroup). Used for **Stage 2/3** co‑distillation of the whole family at once. **Subclasses `DistillationMetaArch`** and overrides `forward_backward` to compute the shared teacher once and broadcast its outputs to each student's rank‑subgroup; the "M students" exist across rank‑subgroups, not as M modules on one rank.

The **objective** (`eupe/distill/`) is identical in all stages: per‑teacher adapter heads project student tokens into each teacher's space; teacher tokens are mean/std‑normalized (stats frozen after a 500‑iter warmup); loss = Σ_teacher [ cosine(cls) + 0.9·cosine(patch) + 0.1·smoothL1(patch) ].

### Data flow (one training step, Stage 2/3)
1. Sampler yields a batch (LVD‑1689M with 10% ImageNet‑1k homogeneous batches).
2. Proxy (frozen) runs once on the global batch → `{cls, patch}`; outputs **broadcast to each student subgroup** (`broadcast_to_subgroups`), downsampled to student resolution if needed (`teacher_to_student_resolution_scale`).
3. Each student runs on its subgroup's slice → `{cls, patch}`.
4. Per student: adapter heads map student→teacher space; teacher tokens normalized; `DistillationLoss` computed; `backprop_loss`; optimizer/scheduler step.
5. Periodic FSDP checkpoint (saved with `teacher.`‑prefixed keys so existing `build_model_for_eval`/`torch.hub` loaders consume them).

---

## 3. Module specifications

### 3.1 `eupe/distill/` — the objective (new package)

| File | Public API (signatures only) | Responsibility |
|---|---|---|
| `__init__.py` | re‑exports | package surface |
| `teachers.py` | `class TeacherModel(ABC)`: `forward(img)->dict{cls,patch}`, `native_resolution: int`, `embed_dim: int`; `class PECoreTeacher/PELangTeacher/DINOv3Teacher/ProxyTeacher(TeacherModel)`; `build_teachers(cfg) -> dict[str,TeacherModel]` | Frozen teacher interface + concrete loaders (real loading = TODO pointing at perception_models / dinov3 / proxy checkpoint). Each runs at its native res (PE 448, DINOv3‑H+ 256). |
| `adapters.py` | `class AdapterHead(nn.Module)(in_dim,hidden_dim,out_dim)`: `forward(x)`; `class AdapterHeadSet(nn.Module)(student_dim, teacher_specs, hidden_dim)`: `forward(cls,patch)->dict` | 2‑layer MLP: `Linear(no bias)→LayerNorm→GELU→Linear(no bias)`. One head per (teacher × {cls,patch}). hidden=1536 (S1) / 3072 (S2‑3). Discarded at eval. |
| `normalize.py` | `class FeatureNormalizer(nn.Module)(dim)` buffers `mean,std`: `forward(x)`, `set_stats(mean,std)`; `estimate_teacher_statistics(teachers, loader, n_iters=500) -> dict` | Per‑teacher, per‑token‑type standardization `(x-mean)/std`. Stats estimated once over ~500 iters, then frozen. Simpler than RADIO PHI‑S (no rotation), avoids per‑step all‑gather. |
| `loss.py` | `class DistillationLoss(nn.Module)(alpha=0.9,beta=0.1,dinov3_patch_gamma=1.0)`: `forward(student_cls,student_patch,teacher_outputs,adapters,normalizer)->dict`; helpers `cosine_loss(a,b)`, `patch_loss(z,y)`, `interpolate_patch_tokens(z,y)` | cosine(cls); α·cos+β·smoothL1(patch); bicubic 2D‑interp smaller patch grid to `max(N_S,N_T)`; Σ over teachers; optional γ on DINOv3 patch term (paper Eq. 4–7). |

### 3.2 `eupe/train/` — the trainer (new package, mirrors `dinov3/train/`)

| File | Public API (signatures only) | Responsibility |
|---|---|---|
| `__init__.py` | re‑exports | package surface |
| `distill_meta_arch.py` | `class DistillationMetaArch(nn.Module)(cfg)`: `forward_backward(data,*,iteration)->loss_dict`, `get_teacher_outputs(images)`, `compute_losses(...)`, `backprop_loss(loss)`, `build_adapters()`, `init_normalizer(loader)` | N frozen teachers → 1 student. Holds student, teachers, adapters, normalizer, loss. Stage‑1 engine; per‑student unit for 2/3. |
| `multidist_meta_arch.py` | `class MultiDistillationMetaArch(DistillationMetaArch)`: `forward_backward(...)`, `broadcast_to_subgroups(x,*,global_batch_size,over_dim)`, `get_teacher_output(images,*,global_batch_size)` | Subclass of `DistillationMetaArch`. Shared proxy computed once, broadcast to each rank‑subgroup; the student local to this rank backprops independently. Mirrors DINOv3 `multidist_meta_arch.py` structure (DINO/iBOT/Sinkhorn losses swapped for `DistillationLoss`). |
| `param_groups.py` | `get_params_groups_with_decay(model, lr, wd, *, layerwise_decay, patch_embed_lr_mult)`, `fuse_params_groups(groups)` | AdamW param groups: layer‑wise LR decay, `patch_embed_lr_mult=0.2`, head WD multiplier. |
| `cosine_lr_scheduler.py` | `class CosineScheduler(...)`: `__getitem__(it)`; `linear_warmup_cosine_decay(...)` | LR/WD/momentum cosine schedules with warmup; supports schedules‑v2 (set peak LR directly to bypass `sqrt_wrt_1024`). |
| `train.py` | `main(args)`, `do_train(cfg, model)`, `build_optimizer(cfg,groups)`, `build_schedulers(cfg)`, `build_data_loader(cfg)`, `apply_optim_scheduler(...)`, `save_checkpoint(...)` | Entry point. Builds model on meta device, FSDP‑wraps, runs normalizer warmup, iterates sampler, logs/checkpoints. Routes to `MultiDistillationMetaArch` when `multidistillation.enabled`. |

### 3.3 `eupe/fsdp/` — sharding (new package, mirrors `dinov3/fsdp/`)

| File | Public API | Responsibility |
|---|---|---|
| `__init__.py` | `get_fsdp_wrapper(...)`, `get_fsdp_modules(...)`, `reshard_fsdp_model(...)` | FSDP helper surface. |
| `ac_compile_parallelize.py` | `parallelize(model, cfg)`, `apply_activation_checkpointing(model)`, `apply_compile(model)` | FSDP(`SHARD_GRAD_OP`, bf16 params / fp32 reduce) + activation checkpointing + `torch.compile`, in one pass. |

### 3.4 `eupe/data/distillation_loaders.py` (new module)

| Public API | Responsibility |
|---|---|
| `make_distillation_data_loader(cfg) -> Iterable`; `MixedSampler(lvd_dataset, imagenet_dataset, imagenet_prob=0.10)`; `build_pyramid_collate(scales)` | Mixed LVD‑1689M + ImageNet‑1k sampler (P(IN1k)=0.10, homogeneous IN1k batches vs heterogeneous LVD batches). Stage‑3 multi‑resolution pyramid collate {256,384,512}. Reuses existing `eupe/data/{loaders,samplers}.py`. |

### 3.5 `DISTILLATION.md` (repo root, new)

Launch commands for all three stages (`python -m eupe.run.submit eupe/train/train.py --nodes N --multi-distillation …`), the fill‑in checklist (which `NotImplementedError`s to implement and the reference file for each), and the validation milestones from the report (proxy Table 4 → Stage 1&2 Table 2 → final Table 1).

---

## 4. Config schema (concrete YAMLs)

All configs merge over the existing `eupe/configs/ssl_default_config.yaml` (already present), which carries `compute_precision`, `student`, `teacher`, `optim`, `crops`, `distillation`, `multidistillation`.

**New `distill:` block — single source of truth for the EUPE objective** (introduced by this scaffold to avoid colliding with DINOv3's legacy single‑teacher `distillation:` block, which EUPE supersedes and leaves unused):
```yaml
distill:
  teachers:                      # list ⇒ Stage 1 (multi-teacher); single-element ⇒ Stage 2/3 (proxy)
    - {name: pecore_g,  config: teachers/pecore_g.yaml}
    - {name: pelang_g,  config: teachers/pelang_g.yaml}
    - {name: dinov3_hplus, config: teachers/dinov3_hplus.yaml}
  adapter_hidden_dim: 1536       # 3072 in Stage 2/3
  normalizer_warmup_iters: 500
  loss: {alpha: 0.9, beta: 0.1, dinov3_patch_gamma: 1.0}
```
The legacy `multidistillation:` block is retained **only** for the rank‑partition launcher (`global_batch_size`, `students[].ranks_range`); it does not configure the objective.

- **`stage1_multiteacher_proxy.yaml`** — `MODEL.META_ARCHITECTURE: DistillationMetaArch`; `student.arch: vit_giant2` (proxy, `n_storage_tokens: 4`); `distill.teachers: [pecore_g, pelang_g, dinov3_hplus]`; `distill.adapter_hidden_dim: 1536`; AdamW, `scaling_rule: sqrt_wrt_1024`, `clip_grad: 3.0`.
- **`stage2_multidistill.yaml`** — `MODEL.META_ARCHITECTURE: MultiDistillationMetaArch`; `multidistillation.{enabled: true, global_batch_size: 8192, students:[…ranks_range…]}`; `distill.teachers: [{name: proxy, config: proxy/vitg_p16.yaml, checkpoint: <STAGE1_CKPT>}]` (single); `distill.adapter_hidden_dim: 3072`; `crops.global_crops_size: 256`; LR target 2e‑5 cosine; wd 1e‑4; **390k iters**.
- **`stage3_multidistill.yaml`** — like Stage 2 but `crops.global_crops_size: [256,384,512]` (pyramid), `global_batch_size: 4096`, LR target 1e‑5, **100k iters**; students init from Stage‑2 (`student.pretrained_weights`).
- **`proxy/vitg_p16.yaml`** — ViT‑G proxy arch (`vit_giant2`, patch16, RoPE base100, 4 reg tokens).
- **`students/{vitt,vits,vitb}_p16.yaml`** — per the repo's `hub/backbones.py` (dims/depth/heads, 4 reg tokens, layerscale 1e‑5, `layernormbf16`, `mask_k_bias`).
- **`students/convnext_{tiny,small,base}_p16.yaml`** — depths/dims from `convnext_sizes`.
- **`teachers/{pecore_g,pelang_g,dinov3_hplus}.yaml`** — `name`, `loader`, `checkpoint`/`hf_id`, `native_resolution` (448/448/256), `embed_dim`.

---

## 5. Risks & mitigations
- **Stub drift from real impl** → each docstring names the exact source file/equation to copy, so bodies are unambiguous.
- **LR scaling ambiguity (`sqrt_wrt_1024`)** → configs default to schedules‑v2 peak‑LR so reproducers hit 2e‑5/1e‑5 regardless of GPU count; documented in `DISTILLATION.md`.
- **Proxy size (1.9B "ViT‑G")** → use `vit_giant2`; docstring/README note to tune FFN/depth to ~1.9B and the 7B path via `vit_7b`.
- **Objective vs DINOv3** → `multidist_meta_arch.py` docstrings explicitly say to keep DINOv3's FSDP/subgroup plumbing but replace DINO/iBOT/Sinkhorn losses with `DistillationLoss`.

---

## 6. File manifest (28 new files)
```
eupe/distill/{__init__,teachers,adapters,normalize,loss}.py            (5)
eupe/train/{__init__,distill_meta_arch,multidist_meta_arch,param_groups,cosine_lr_scheduler,train}.py  (6)
eupe/fsdp/{__init__,ac_compile_parallelize}.py                          (2)
eupe/data/distillation_loaders.py                                       (1)
eupe/configs/train/{stage1_multiteacher_proxy,stage2_multidistill,stage3_multidistill}.yaml  (3)
eupe/configs/train/proxy/vitg_p16.yaml                                  (1)
eupe/configs/train/students/{vitt,vits,vitb,convnext_tiny,convnext_small,convnext_base}_p16.yaml  (6)
eupe/configs/train/teachers/{pecore_g,pelang_g,dinov3_hplus}.yaml       (3)
DISTILLATION.md                                                         (1)
```
(Existing files unchanged. `eupe/run/submit.py` + `eupe/configs/config.py::setup_multidistillation` already support the `--multi-distillation` launch path.)
