import numpy as np
import pytest

from core.geometry import Rect
from core.static_veto import StaticVeto


def textured(seed=3, size=(190, 230)):
    return np.random.default_rng(seed).integers(60, 120, size, dtype=np.uint8)


def with_blob(base, x, y):
    img = base.copy()
    img[y:y + 30, x:x + 14] = 230
    img[y + 8:y + 22, x + 4:x + 10] = 40               # internal structure so correlation is meaningful
    return img


def template_from(img, x, y, pad=0):
    return img[y - pad:y + 30 + pad, x - pad:x + 14 + pad].copy()


BOX = Rect(100, 80, 14, 30)


def feed(veto, frames, step=0.1, t0=0.0, origin=(0, 0)):
    t = t0
    for f in frames:
        veto.observe(f, origin, t=t)
        t += step
    return t


def test_static_object_is_vetoed():
    base = textured()
    blob = with_blob(base, 100, 80)
    veto = StaticVeto()
    now = feed(veto, [blob] * 30)                               # 3 s of the same scene
    r = veto.is_static(BOX, 0.83, template_from(blob, 100, 80), t=now)
    assert r.static and r.ref_score > 0.95 and r.ref_age >= 0.8


def test_newly_appeared_object_is_NOT_vetoed():
    base = textured()
    veto = StaticVeto()
    now = feed(veto, [base] * 30)                               # empty scene for 3 s
    blob = with_blob(base, 100, 80)
    veto.observe(blob, t=now)
    r = veto.is_static(BOX, 0.95, template_from(blob, 100, 80), t=now)
    assert not r.static and r.ref_score < 0.5


def test_a_real_bang_that_is_only_0p5s_old_is_not_vetoed():
    base = textured()
    blob = with_blob(base, 100, 80)
    veto = StaticVeto(min_age_s=0.8)
    t = feed(veto, [base] * 25)                                 # 2.5 s empty
    t = feed(veto, [blob] * 5, t0=t)                            # '!' visible for 0.5 s: younger than min_age
    r = veto.is_static(BOX, 0.9, template_from(blob, 100, 80), t=t)
    assert not r.static


def test_no_history_means_no_veto():
    veto = StaticVeto()
    blob = with_blob(textured(), 100, 80)
    veto.observe(blob, t=0.0)
    r = veto.is_static(BOX, 0.9, template_from(blob, 100, 80), t=0.2)   # only a 0.2 s old frame
    assert r == r.__class__(False, None, None)


def test_shimmer_that_crossed_the_threshold_is_vetoed():
    """Score hovers 0.70 -> 0.74 at the same place: the exact pattern of the morning false triggers."""
    base = textured()
    blob = with_blob(base, 100, 80)
    noisy = np.clip(blob.astype(int) + np.random.default_rng(9).integers(-6, 7, blob.shape), 0, 255).astype(np.uint8)
    veto = StaticVeto()
    now = feed(veto, [noisy] * 30)
    r = veto.is_static(BOX, 0.74, template_from(blob, 100, 80), t=now)
    assert r.static


def test_camera_move_gives_no_evidence_instead_of_a_wrong_veto():
    base = textured()
    blob = with_blob(base, 100, 80)
    veto = StaticVeto()
    now = feed(veto, [blob] * 30, origin=(400, 0))              # old frames cover screen x 400..630
    r = veto.is_static(BOX, 0.83, template_from(blob, 100, 80), t=now)   # hit at screen x 100: outside them
    assert not r.static and r.ref_score is None


def test_history_is_bounded():
    veto = StaticVeto(history_s=3.0)
    feed(veto, [textured()] * 200, step=0.1)
    assert len(veto._frames) <= 31


def test_reset_forgets():
    veto = StaticVeto()
    now = feed(veto, [textured()] * 30)
    veto.reset()
    assert veto.is_static(BOX, 0.8, np.zeros((30, 14), np.uint8), t=now).ref_score is None
