import time

import cv2
import numpy as np
import pytest

from core.progress_watchdog import ProgressWatchdog
from core.region_scan import RegionScanner, ScanTarget


def make_frame(seed=0, size=(1080, 1920)):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, size, dtype=np.uint8), (0, 0), 2)


def paste(frame, tpl, x, y):
    f = frame.copy()
    f[y:y + tpl.shape[0], x:x + tpl.shape[1]] = tpl
    return f


def tpl_image(seed, size=(60, 200)):
    return cv2.GaussianBlur(np.random.default_rng(seed).integers(0, 255, size, dtype=np.uint8), (0, 0), 1.5)


@pytest.fixture()
def scanner(tmp_path):
    for name, seed in (("repair", 11), ("ok", 12), ("cast", 13)):
        d = tmp_path / name
        d.mkdir()
        cv2.imwrite(str(d / f"{name}.png"), tpl_image(seed))
    return RegionScanner([
        ScanTarget("repair", str(tmp_path / "repair"), (0.2, 0.1, 0.8, 0.9), 0.8),
        ScanTarget("ok", str(tmp_path / "ok"), (0.2, 0.4, 0.8, 0.9), 0.8),
        ScanTarget("cast", str(tmp_path / "cast"), (0.65, 0.55, 1.0, 1.0), 0.8),
    ])


def test_finds_templates_inside_their_regions(scanner):
    f = paste(make_frame(), tpl_image(11), 831, 707)            # repair modal, central
    hits = scanner.scan(f)
    assert hits["repair"].found and hits["repair"].score > 0.99
    assert (hits["repair"].x, hits["repair"].y) == (831, 707)
    assert not hits["ok"].found and not hits["cast"].found


def test_template_outside_its_region_is_ignored(scanner):
    f = paste(make_frame(), tpl_image(13), 100, 100)            # a "cast button" in the top-left corner
    assert not scanner.scan(f)["cast"].found


def test_coordinates_include_the_capture_origin(scanner):
    f = paste(make_frame(), tpl_image(13), 1500, 800)
    h = scanner.scan(f, origin=(1920, 0))["cast"]
    assert h.found and (h.x, h.y) == (1920 + 1500, 800)


def test_names_subset_only_scans_those(scanner):
    assert set(scanner.scan(make_frame(), names=["ok"])) == {"ok"}


def test_first_found_respects_priority_and_stops(scanner):
    f = paste(paste(make_frame(), tpl_image(12), 700, 600), tpl_image(13), 1500, 800)
    assert scanner.first_found(f, ["repair", "ok", "cast"])[0] == "ok"
    assert scanner.first_found(f, ["cast", "ok"])[0] == "cast"
    assert scanner.first_found(make_frame(), ["repair", "ok", "cast"]) is None


def test_context_check_is_far_below_the_old_seven_seconds(scanner):
    f = make_frame()
    scanner.scan(f)
    t = time.perf_counter()
    for _ in range(3):
        scanner.scan(f)
    assert (time.perf_counter() - t) / 3 < 0.6          # old pattern: ~7 s for six full-frame multi-scale calls


def test_missing_templates_fail_fast(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="repair"):
        RegionScanner([ScanTarget("repair", str(tmp_path / "empty"), (0, 0, 1, 1), 0.8)])


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_watchdog_stalls_only_after_timeout_and_progress_resets_it():
    c = Clock()
    w = ProgressWatchdog(timeout_s=90, clock=c)
    c.t = 89
    assert not w.stalled()
    c.t = 91
    assert w.stalled() and w.idle_s() == pytest.approx(91)
    w.note_progress("cast click")
    assert not w.stalled() and w.last_event == "cast click"
    c.t = 91 + 89
    assert not w.stalled()


def test_watchdog_warns_once_per_stall_and_rearms_on_progress():
    c = Clock()
    w = ProgressWatchdog(timeout_s=90, clock=c)
    c.t = 20
    assert not w.warn_due(30)
    c.t = 31
    assert w.warn_due(30) is True
    assert w.warn_due(30) is False                      # only once
    w.note_progress("cast")
    c.t = 31 + 31
    assert w.warn_due(30) is True                       # re-armed by progress
