# Multi-Stage Distillation — Body Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan FILLS IN the `NotImplementedError` stubs created by the scaffold plan (`2026-05-29-multistage-multiteacher-distillation.md`); it does not create new files.

**Goal:** Replace every `NotImplementedError` body in `eupe/{distill,train,fsdp,data}` with a faithful, working implementation ported from the cloned reference repos + the paper recipe, so the 3-stage pipeline runs once a team supplies GPUs + data + teacher checkpoints.

**Architecture:** A DINOv3-style trainer (`eupe/train/`) carrying a RADIO-style feature-matching objective (`eupe/distill/`). `DistillationMetaArch` = N frozen teachers → 1 student (Stage 1). `MultiDistillationMetaArch` subclasses it for 1 proxy → M students on rank-subgroups (Stage 2/3). FSDP in `eupe/fsdp/`, mixed LVD+IN1k data in `eupe/data/distillation_loaders.py`.

**Tech Stack:** Python 3.11, PyTorch ≥ 2.7.1, FSDP, OmegaConf, submitit. Reference repos cloned at `/Users/puyangwang/EUPE/refs/{dinov3,RADIO,perception_models}` (siblings of the repo; NOT inside its git tree).

**Conventions (do not deviate):**
- Keep the existing FAIR license header already at the top of each stub file. Use `logging.getLogger("eupe")`.
- Keep each stub's existing public signatures (other modules import them). Implement bodies only; add private helpers as needed.
- Read `cfg.distill.*` for objective config (NOT the legacy `cfg.distillation.*`, which EUPE leaves unused — see spec §4).
- Every ported B/C body gets a one-line provenance comment: `# Ported from <refs/path>:<symbol> — <divergence>`.
- Mark any genuinely-inferred external API (where the cloned source is ambiguous) with `# RECONSTRUCTED (unverified): <what/why>`.

---

## Reference source map (port from these)

| Target stub | Port from |
|---|---|
| `distill/adapters.py` | `refs/RADIO/radio/adaptor_mlp.py::MLP2` (structure) + paper §4.1 (drop biases) |
| `distill/normalize.py` | `refs/RADIO/radio/feature_normalizer.py::FeatureNormalizer` (drop rotation `tx`) + report §6.3 |
| `distill/loss.py` | AM-RADIO objective + report §6.2 |
| `distill/teachers.py` | `refs/perception_models/core/vision_encoder/pe.py` (PE-core/lang), dinov3 hub (DINOv3-H+), `eupe.models.build_model` (proxy) |
| `train/param_groups.py` | `refs/dinov3/dinov3/train/param_groups.py` |
| `train/cosine_lr_scheduler.py` | `refs/dinov3/dinov3/train/cosine_lr_scheduler.py` |
| `train/distill_meta_arch.py` | `refs/dinov3/dinov3/train/ssl_meta_arch.py` (strip DINO/iBOT/Sinkhorn; insert `DistillationLoss`) |
| `train/multidist_meta_arch.py` | `refs/dinov3/dinov3/train/multidist_meta_arch.py` (keep subgroup broadcast; swap loss) |
| `train/train.py` | `refs/dinov3/dinov3/train/train.py` (loop/checkpoint; normalizer warmup; `distill:` routing) |
| `fsdp/ac_compile_parallelize.py` | `refs/dinov3/dinov3/fsdp/ac_compile_parallelize.py` |
| `data/distillation_loaders.py` | existing `eupe/data/{loaders,samplers,transforms}.py` + dinov3 IN1k/LVD interleave |

---

## Frozen interfaces (the contract — keep these EXACT so parallel work does not drift)

**`distill/adapters.py`**
- `AdapterHead(in_dim, hidden_dim, out_dim)`: `fc1=Linear(in_dim,hidden_dim,bias=False)`, `norm=LayerNorm(hidden_dim)`, `act=GELU()`, `fc2=Linear(hidden_dim,out_dim,bias=False)`. `forward(x)` = `fc2(act(norm(fc1(x))))`; works for `[B,d]` and `[B,N,d]`.
- `AdapterHeadSet(student_dim, teacher_specs: Sequence[(name,dim)], hidden_dim)`: `nn.ModuleDict` `name → ModuleDict({"cls":AdapterHead, "patch":AdapterHead})`. `forward(student_cls[B,dS], student_patch[B,NS,dS]) → {name:{"cls":[B,dT], "patch":[B,NS,dT]}}`.

**`distill/normalize.py`**
- `FeatureNormalizer(dim)`: `register_buffer("mean", zeros(dim))`, `register_buffer("std", ones(dim))`. `set_stats(mean,std)` → `copy_` into buffers, `std.clamp_(min=1e-6)`. `forward(x) = (x - mean) / std` (broadcast over leading dims).
- `estimate_teacher_statistics(teachers: Dict[str,nn.Module], data_loader, n_iters=500) → Dict[name → {"cls":FeatureNormalizer, "patch":FeatureNormalizer}]`. Streaming fp32 sum/sumsq over `n_iters` batches under `torch.no_grad()`; all-reduce across ranks if `distributed.is_enabled()`; `mean=sum/count`, `std=sqrt(max(sumsq/count - mean², 0))`; build + `set_stats` each normalizer.

**`distill/loss.py`**
- `DistillationLoss(alpha=0.9, beta=0.1, dinov3_patch_gamma=1.0, dinov3_teacher_name="dinov3_hplus")`.
- `@staticmethod cosine_loss(z, y) → mean(1 - F.cosine_similarity(z, y, dim=-1))`.
- `@staticmethod interpolate_patch_tokens(z[B,NS,d], y[B,NT,d]) → (z',y')` both at `max(NS,NT)` tokens via square-grid bicubic interp of the smaller grid.
- `patch_loss(z, y) → alpha*cosine_loss(z,y) + beta*F.smooth_l1_loss(z, y)` after `interpolate_patch_tokens`.
- `forward(adapted_student: {name:{cls,patch}}, teacher_normalized: {name:{cls,patch}}) → {"loss": scalar, "<name>_cls": float-tensor, "<name>_patch": float-tensor}`. Per teacher: `Li_c = cosine_loss(cls)`, `Li_p = patch_loss(patch)` (×γ if name == dinov3_teacher_name); `loss = Σ(Li_c + Li_p)`.

**`distill/teachers.py`** — `TeacherModel(ABC, nn.Module)` with attrs `native_resolution:int`, `embed_dim:int`, `forward(img)→{"cls":[B,d],"patch":[B,N,d]}`. `PECoreTeacher/PELangTeacher/DINOv3Teacher/ProxyTeacher`. `build_teachers(cfg)→Dict[str,TeacherModel]` (cuda, eval, requires_grad_(False)).

**`train/distill_meta_arch.py`** — `DistillationMetaArch(cfg)` holds `self.student`, `self.teachers` (ModuleDict, frozen), `self.adapters` (AdapterHeadSet), `self.normalizers` (ModuleDict name→ModuleDict{cls,patch}), `self.loss`. Methods: `init_normalizer(loader)`, `get_teacher_outputs(images)→{name:{cls,patch}}`, `compute_losses(student_cls, student_patch, teacher_outputs)→loss_dict`, `backprop_loss(loss)`, `forward_backward(data, *, iteration=0, **ignored)→loss_dict`.

**`train/multidist_meta_arch.py`** — `MultiDistillationMetaArch(DistillationMetaArch)`: `broadcast_to_subgroups(x, *, global_batch_size, over_dim=0)`, `get_teacher_output(images, *, global_batch_size)→{name:{cls,patch}}`, `forward_backward(...)`.

**`train/param_groups.py`** — `get_params_groups_with_decay(model, lr, wd, *, layerwise_decay=1.0, patch_embed_lr_mult=0.2) → List[dict]`; `fuse_params_groups(groups) → List[dict]`.

**`train/cosine_lr_scheduler.py`** — `CosineScheduler(base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0.0, freeze_iters=0)` with precomputed `self.schedule` (numpy) and `__getitem__(it)→float`.

**`train/train.py`** — `get_args_parser`, `build_optimizer(cfg, groups)`, `build_schedulers(cfg)→dict`, `apply_optim_scheduler(opt, scheds, it)`, `build_data_loader(cfg)`, `do_train(cfg, model)`, `main(args=None)`.

**`fsdp/ac_compile_parallelize.py`** — `apply_activation_checkpointing(model, full=False)`, `apply_compile(model)`, `parallelize(model, cfg)`.

**`data/distillation_loaders.py`** — `MixedSampler(lvd_dataset, imagenet_dataset, imagenet_prob=0.10)`, `build_pyramid_collate(scales:List[int])`, `make_distillation_data_loader(cfg)→Iterable`.

---

## Track A — Objective (CPU-testable; full TDD, run here)

Tests live in `tests/distill/`. Run with `python -m pytest tests/distill -q` (verify stage will `pip install pytest` if missing). Tiny CPU tensors only.

### Task A1: `distill/adapters.py`

**Files:** Modify `eupe/distill/adapters.py`; Create `tests/distill/test_adapters.py`.

- [ ] **Step 1: Write failing test**
```python
# tests/distill/test_adapters.py
import torch
from eupe.distill.adapters import AdapterHead, AdapterHeadSet

def test_adapter_head_structure_and_shapes():
    h = AdapterHead(16, 32, 24)
    assert h.fc1.bias is None and h.fc2.bias is None          # no-bias (paper §4.1)
    assert isinstance(h.norm, torch.nn.LayerNorm)
    assert h(torch.randn(4, 16)).shape == (4, 24)             # cls path
    assert h(torch.randn(4, 7, 16)).shape == (4, 7, 24)       # patch path

def test_adapter_head_set_routes_per_teacher():
    s = AdapterHeadSet(student_dim=16, teacher_specs=[("t_a", 24), ("t_b", 40)], hidden_dim=32)
    out = s(torch.randn(2, 16), torch.randn(2, 5, 16))
    assert set(out) == {"t_a", "t_b"}
    assert out["t_a"]["cls"].shape == (2, 24) and out["t_a"]["patch"].shape == (2, 5, 24)
    assert out["t_b"]["cls"].shape == (2, 40) and out["t_b"]["patch"].shape == (2, 5, 40)
```
- [ ] **Step 2: Run, expect FAIL** (`NotImplementedError`). `python -m pytest tests/distill/test_adapters.py -q`
- [ ] **Step 3: Implement** `AdapterHead.__init__/forward` and `AdapterHeadSet.__init__/forward` per the frozen interface. Lineage: `refs/RADIO/radio/adaptor_mlp.py::MLP2` (num_inner=0), biases removed.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(distill): implement adapter heads (paper §4.1)`

### Task A2: `distill/normalize.py`

**Files:** Modify `eupe/distill/normalize.py`; Create `tests/distill/test_normalize.py`.

- [ ] **Step 1: Write failing test**
```python
# tests/distill/test_normalize.py
import torch
from eupe.distill.normalize import FeatureNormalizer, estimate_teacher_statistics

def test_normalizer_buffers_frozen_and_standardize():
    n = FeatureNormalizer(8)
    assert not any(p.requires_grad for p in n.parameters())   # only buffers, no params
    torch.testing.assert_close(n(torch.zeros(3, 8)), torch.zeros(3, 8))   # mean0/std1 identity
    n.set_stats(torch.full((8,), 2.0), torch.full((8,), 4.0))
    torch.testing.assert_close(n(torch.full((3, 8), 6.0)), torch.ones(3, 8))  # (6-2)/4

def test_set_stats_clamps_zero_std():
    n = FeatureNormalizer(4); n.set_stats(torch.zeros(4), torch.zeros(4))
    assert torch.all(n.std > 0)

def test_estimate_statistics_recovers_distribution():
    torch.manual_seed(0)
    mu, sd = 5.0, 3.0
    class DummyTeacher(torch.nn.Module):
        embed_dim = 8
        def forward(self, img):
            b = img.shape[0]
            return {"cls": mu + sd*torch.randn(b, 8), "patch": mu + sd*torch.randn(b, 6, 8)}
    loader = ([torch.randn(64, 3, 16, 16)] for _ in range(200))   # img content irrelevant to dummy
    norms = estimate_teacher_statistics({"t": DummyTeacher()}, loader, n_iters=200)
    assert torch.allclose(norms["t"]["cls"].mean, torch.full((8,), mu), atol=0.3)
    assert torch.allclose(norms["t"]["cls"].std,  torch.full((8,), sd), atol=0.3)
```
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `FeatureNormalizer` (+`set_stats`,`forward`) and `estimate_teacher_statistics` (streaming fp32 sum/sumsq; all-reduce guarded by `distributed.is_enabled()`; per `cls`/`patch`, flattening patch over tokens). Provenance: simplified `refs/RADIO/radio/feature_normalizer.py` (no rotation).
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(distill): implement frozen mean/std normalizer + 500-iter estimator`

### Task A3: `distill/loss.py`

**Files:** Modify `eupe/distill/loss.py`; Create `tests/distill/test_loss.py`.

- [ ] **Step 1: Write failing test**
```python
# tests/distill/test_loss.py
import torch
from eupe.distill.loss import DistillationLoss

def test_cosine_loss_bounds():
    v = torch.randn(4, 10)
    assert torch.allclose(DistillationLoss.cosine_loss(v, v), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(DistillationLoss.cosine_loss(v, -v), torch.tensor(2.0), atol=1e-6)

def test_interpolate_aligns_to_larger_grid():
    z = torch.randn(2, 4, 5)    # 2x2 grid
    y = torch.randn(2, 16, 5)   # 4x4 grid
    z2, y2 = DistillationLoss.interpolate_patch_tokens(z, y)
    assert z2.shape == (2, 16, 5) and y2.shape == (2, 16, 5)

def test_forward_assembles_loss_dict_and_applies_gamma():
    loss = DistillationLoss(alpha=0.9, beta=0.1, dinov3_patch_gamma=2.0, dinov3_teacher_name="d")
    adapted = {"d": {"cls": torch.randn(2, 6), "patch": torch.randn(2, 9, 6)}}
    tnorm   = {"d": {"cls": torch.randn(2, 6), "patch": torch.randn(2, 9, 6)}}
    out = loss(adapted, tnorm)
    assert "loss" in out and "d_cls" in out and "d_patch" in out
    assert out["loss"].requires_grad is False or out["loss"].dim() == 0
    # gamma path: doubling gamma doubles the patch contribution
    base = DistillationLoss(0.9, 0.1, 1.0, "d")(adapted, tnorm)
    torch.testing.assert_close(out["d_patch"], 2.0 * base["d_patch"])
```
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `cosine_loss`, `interpolate_patch_tokens` (reshape `[B,N,d]→[B,d,√N,√N]`, `F.interpolate(mode="bicubic", align_corners=False)`, flatten back), `patch_loss`, `forward` (Σ over teachers, γ on dinov3 patch). Report §6.2.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(distill): implement multi-teacher feature-matching loss (§6.2)`

### Task A4: verify objective package wiring

**Files:** none (package `__init__.py` already exports). 
- [ ] **Step 1** `python -c "from eupe.distill import AdapterHeadSet, FeatureNormalizer, estimate_teacher_statistics, DistillationLoss, TeacherModel, build_teachers"` → exit 0 (after B1 lands `teachers`; if run before, expect only teachers-import errors).
- [ ] **Step 2** `python -m pytest tests/distill -q` → all pass.
- [ ] **Step 3: Commit** if any wiring fix needed.

---

## Track B — Teachers & Data (no run env; verify by py_compile + diff vs upstream)

### Task B1: `distill/teachers.py`

**Files:** Modify `eupe/distill/teachers.py`.

- [ ] **Step 1: Implement teacher wrappers.**
  - `DINOv3Teacher`: load via dinov3 (`torch.hub.load('facebookresearch/dinov3', ...)` or local checkpoint into a dinov3 ViT). `forward`: `out = model.forward_features(img); return {"cls": out["x_norm_clstoken"], "patch": out["x_norm_patchtokens"]}`. `embed_dim=1280`, `native_resolution=256`. (Green — same dict as EUPE ViT.)
  - `PECoreTeacher`/`PELangTeacher`: load via `perception_models/core/vision_encoder/pe.py` (`VisionTransformer.from_config(...)` then load checkpoint). `forward`: `tokens = model.forward_features(img); cls = model._pool(tokens); patch = tokens[:, int(model.use_cls_token):]`. `embed_dim` from config (PE-G width), `native_resolution=448`. Mark `# RECONSTRUCTED (unverified): PE pooling/token-strip API` if the from_config path is uncertain.
  - `ProxyTeacher`: `eupe.models.build_model(cfg.student, only_teacher=True, img_size=native_resolution)`; load `teacher.`-prefixed checkpoint (mirror `eupe/models/__init__.build_model_for_eval`); `forward` via `forward_features` dict. (Green.)
  - `build_teachers(cfg)`: iterate `cfg.distill.teachers`, look up `_TEACHER_REGISTRY`, instantiate from each entry's `config`/`checkpoint`, `.cuda().eval().requires_grad_(False)`. Single `proxy` entry ⇒ Stage 2/3.
  - Provenance comment per loader citing the refs path.
- [ ] **Step 2: Verify** `python -m py_compile eupe/distill/teachers.py` → exit 0.
- [ ] **Step 3: Verify** `python -c "import ast; ast.parse(open('eupe/distill/teachers.py').read())"` and confirm every method body is non-`NotImplementedError`.
- [ ] **Step 4: Commit** `feat(distill): implement frozen teacher loaders (PE/DINOv3/proxy)`

### Task B2: `data/distillation_loaders.py`

**Files:** Modify `eupe/data/distillation_loaders.py`. First READ `eupe/data/{loaders.py,samplers.py,transforms.py}` to reuse `SamplerType`, dataset parsing, and transform builders.

- [ ] **Step 1: Implement** `MixedSampler` (Bernoulli(`imagenet_prob`) per batch → homogeneous IN1k vs heterogeneous LVD; infinite), `build_pyramid_collate(scales)` (per-sample random scale resize → pad/stack, Stage 3), `make_distillation_data_loader(cfg)` (build the two datasets from `cfg.train.dataset_path`, transforms from `cfg.crops` incl. RRC/flip/jitter/blur/solarize, `MixedSampler`, `DataLoader(batch_size_per_gpu, num_workers)`; attach pyramid collate iff `cfg.crops.global_crops_size` is a list). Provenance: reuse existing eupe data modules + dinov3 IN1k/LVD interleave (P=0.10).
- [ ] **Step 2: Verify** `python -m py_compile eupe/data/distillation_loaders.py` → exit 0.
- [ ] **Step 3: Commit** `feat(data): implement LVD+IN1k mixed sampler + pyramid collate`

---

## Track C — Trainer & FSDP (depends on A+B interfaces; verify by py_compile + import graph)

### Task C1: `train/param_groups.py`
**Files:** Modify `eupe/train/param_groups.py`. Port from `refs/dinov3/dinov3/train/param_groups.py`.
- [ ] **Step 1: Implement** `get_params_groups_with_decay` (per-param layer index → `lr_mult = layerwise_decay ** (last-idx)`; `patch_embed` → `patch_embed_lr_mult`; `wd=0` for ndim==1 and names containing `token`) and `fuse_params_groups` (merge by `(lr_mult, wd)`). Adapt key names to EUPE block/module naming.
- [ ] **Step 2: Verify** `python -m py_compile eupe/train/param_groups.py`. **Commit** `feat(train): implement AdamW param groups w/ layerwise decay`.

### Task C2: `train/cosine_lr_scheduler.py`
**Files:** Modify `eupe/train/cosine_lr_scheduler.py`. Port from `refs/dinov3/dinov3/train/cosine_lr_scheduler.py`.
- [ ] **Step 1: Implement** `CosineScheduler` precomputing `self.schedule` = concat(freeze, linear warmup, cosine decay) as numpy length `total_iters`; `__getitem__(it)=schedule[min(it,len-1)]`.
- [ ] **Step 2:** Add a quick CPU test `tests/distill/test_scheduler.py`: assert `sched[0]==start_warmup_value`, `sched[warmup_iters]≈base_value`, `sched[-1]≈final_value`, monotonic decay after warmup. Run `python -m pytest tests/distill/test_scheduler.py -q`.
- [ ] **Step 3: Verify + Commit** `feat(train): implement cosine LR/WD/momentum scheduler`.

### Task C3: `fsdp/ac_compile_parallelize.py`
**Files:** Modify `eupe/fsdp/ac_compile_parallelize.py`. Port from `refs/dinov3/dinov3/fsdp/ac_compile_parallelize.py`.
- [ ] **Step 1: Implement** `parallelize` (MixedPrecision from `cfg.compute_precision.{param_dtype,reduce_dtype}`; `ShardingStrategy[cfg.compute_precision.sharding_strategy]`; wrap EUPE transformer blocks; meta→`to_empty`/cuda), `apply_activation_checkpointing` (wrap blocks; `full` flag), `apply_compile` (`torch.compile` per block). Identify EUPE's block class from `eupe/layers/block.py`.
- [ ] **Step 2: Verify** `python -m py_compile eupe/fsdp/ac_compile_parallelize.py` and `python -c "from eupe.fsdp import parallelize, apply_activation_checkpointing, apply_compile"`. **Commit** `feat(fsdp): implement FSDP + activation-ckpt + compile`.

### Task C4: `train/distill_meta_arch.py`
**Files:** Modify `eupe/train/distill_meta_arch.py`. Port from `refs/dinov3/dinov3/train/ssl_meta_arch.py` (strip DINO/iBOT/Sinkhorn/koleo; replace objective with `DistillationLoss`).
- [ ] **Step 1: Implement** `__init__` (student via `eupe.models.build_model_from_cfg(cfg)` keeping the student; `build_teachers(cfg)`; `AdapterHeadSet(student.embed_dim, [(n,t.embed_dim) for n,t in teachers], cfg.distill.adapter_hidden_dim)`; placeholder `FeatureNormalizer`s per teacher×{cls,patch}; `DistillationLoss(**cfg.distill.loss)`), `init_normalizer` (call `estimate_teacher_statistics`, `set_stats`), `get_teacher_outputs` (per-teacher resize to `native_resolution`, `no_grad` forward), `compute_losses` (adapters → normalize teacher tokens → `DistillationLoss`), `backprop_loss` (plain bf16 `loss.backward()`, clip_grad per `cfg.optim.clip_grad`), `forward_backward` (student `forward_features` → cls/patch → teacher outputs → compute_losses → backprop).
- [ ] **Step 2: Verify** `python -m py_compile eupe/train/distill_meta_arch.py` + import. **Commit** `feat(train): implement single-student distillation meta-arch`.

### Task C5: `train/multidist_meta_arch.py`
**Files:** Modify `eupe/train/multidist_meta_arch.py`. Port from `refs/dinov3/dinov3/train/multidist_meta_arch.py`.
- [ ] **Step 1: Implement** `broadcast_to_subgroups` (assemble global-batch teacher output via world all-gather, then slice this subgroup's portion — mirror dinov3), `get_teacher_output` (proxy forward once + broadcast), `forward_backward` (shared-teacher → local student `compute_losses` → backprop). Keep dinov3's subgroup plumbing using `eupe.distributed` primitives; provenance comment noting "identical to dinov3 except `DistillationLoss`".
- [ ] **Step 2: Verify** `python -m py_compile eupe/train/multidist_meta_arch.py` + import. **Commit** `feat(train): implement multi-student co-distillation meta-arch`.

### Task C6: `train/train.py`
**Files:** Modify `eupe/train/train.py`. Port loop/checkpoint from `refs/dinov3/dinov3/train/train.py`.
- [ ] **Step 1: Implement** `get_args_parser`, `build_optimizer` (AdamW betas from cfg), `build_schedulers` (lr/wd/momentum `CosineScheduler`, schedules-v2 peak-aware), `apply_optim_scheduler`, `build_data_loader` (→ `make_distillation_data_loader`), `do_train` (build meta-arch → `parallelize` → `init_normalizer` warmup → iterate: `forward_backward`, schedule, optimizer step, periodic checkpoint with `teacher.`-prefixed keys), `main` (route to `MultiDistillationMetaArch` iff `cfg.multidistillation.enabled`, inside `job_context`).
- [ ] **Step 2: Verify** `python -m py_compile eupe/train/train.py` + `python -c "from eupe.train.train import main"`. **Commit** `feat(train): implement distillation training loop + checkpointing`.

---

## Final verify stage
- [ ] `python -m py_compile $(git ls-files 'eupe/**/*.py')` → no output.
- [ ] `python -m pytest tests/distill -q` → all green.
- [ ] `python -c "import eupe.distill, eupe.train, eupe.fsdp, eupe.data.distillation_loaders"` → exit 0 (imports resolve; runtime deps like `dinov3`/`perception_models` are lazy-imported inside teacher loaders, so this should pass without them).
- [ ] Confirm no `NotImplementedError` remains: `! grep -rn "raise NotImplementedError" eupe/distill eupe/train eupe/fsdp eupe/data/distillation_loaders.py` (the existing `eupe/models/__init__.py` arch guard is allowed; scope grep to the new files).

---

## Self-Review

**1. Spec coverage:** distill ×4 (adapters/normalize/loss/teachers) → A1-A3,B1; train ×6 → C1-C6; fsdp → C3; data → B2. All 11 stub modules covered. Objective math (spec §3.1/§6) → A1-A3. Two-engine arch (spec §2) → C4/C5. `distill:` config consumption (spec §4) → C4/C6. ✓
**2. Placeholder scan:** No "TBD/fill-in later". B/C steps name the exact ref source + adaptation rather than pasting upstream code — a deliberate choice for a port task with sources on disk (stated in the header), not a placeholder. Track A has complete runnable test code. ✓
**3. Type/name consistency:** `AdapterHeadSet`/`FeatureNormalizer`/`estimate_teacher_statistics`/`DistillationLoss`/`build_teachers` defined A1-A3,B1 and consumed in C4; `DistillationMetaArch` (C4) subclassed by `MultiDistillationMetaArch` (C5); `parallelize` (C3) used in C6; `make_distillation_data_loader` (B2) used in C6; `CosineScheduler` (C2) used in C6. Names match the frozen-interface section. ✓
**4. Ambiguity:** ConvNeXt cls/patch resolved (uniform `x_norm_*` dict, no branch). PE token API pinned to `forward_features`+`_pool`. Subgroup broadcast cites the 165-line dinov3 source verbatim. ✓
