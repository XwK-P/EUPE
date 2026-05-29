# tests/distill/test_scheduler.py
import numpy as np

from eupe.train.cosine_lr_scheduler import CosineScheduler


def test_schedule_length_and_endpoints():
    sched = CosineScheduler(
        base_value=1.0,
        final_value=0.1,
        total_iters=100,
        warmup_iters=10,
        start_warmup_value=0.0,
        freeze_iters=0,
    )
    assert len(sched.schedule) == 100
    assert sched[0] == 0.0  # start_warmup_value
    assert np.isclose(sched[10], 1.0)  # base_value at end of warmup
    # Cosine reaches final_value asymptotically (dinov3 divides by len(iters), not len-1),
    # so the last index lands just above final_value; near it within one cosine step.
    assert np.isclose(sched[-1], 0.1, atol=2e-3)
    assert sched[-1] >= 0.1


def test_freeze_holds_start_warmup_value():
    sched = CosineScheduler(
        base_value=2.0,
        final_value=0.5,
        total_iters=50,
        warmup_iters=5,
        start_warmup_value=0.3,
        freeze_iters=8,
    )
    assert len(sched.schedule) == 50
    # All freeze iters held at start_warmup_value.
    for it in range(8):
        assert np.isclose(sched[it], 0.3)
    # End of warmup reaches base_value.
    assert np.isclose(sched[8 + 5], 2.0)
    assert np.isclose(sched[-1], 0.5, atol=2e-2)
    assert sched[-1] >= 0.5


def test_monotonic_cosine_decay_after_warmup():
    sched = CosineScheduler(
        base_value=1.0,
        final_value=0.0,
        total_iters=200,
        warmup_iters=20,
    )
    decay = sched.schedule[20:]
    diffs = np.diff(decay)
    assert np.all(diffs <= 1e-12)  # non-increasing through the cosine portion


def test_getitem_clamps_out_of_range():
    sched = CosineScheduler(base_value=1.0, final_value=0.2, total_iters=30, warmup_iters=3)
    # Out-of-range index clamps to the last schedule entry (schedule[min(it, len-1)]).
    assert sched[10_000] == sched[29]
    assert sched[10_000] == sched.schedule[-1]
