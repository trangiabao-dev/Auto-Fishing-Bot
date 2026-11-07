import time
import numpy as np
import pytest
from core.farm_engine import FarmEngine, FarmState
from core.farm_config_manager import FarmConfigManager, FruitRule

class BlankCapturer:
    def capture(self, region=None):
        # Blank 1080p screen where no buttons exist
        return np.zeros((1080, 1920, 3), dtype=np.uint8), (0, 0)

def test_step1_timeout_watchdog_aborts_loop():
    engine = FarmEngine(capturer=BlankCapturer())
    engine.step1_timeout_s = 0.5  # Fast timeout for test
    logs = []
    engine.log_callback = logs.append

    cfg = FarmConfigManager()
    cfg.rules = {"Hạt dẻ": FruitRule("Hạt dẻ", 3.0, True)}
    engine.cfg_mgr = cfg

    t0 = time.time()
    # Run _harvest_single_fruit directly
    engine._harvest_single_fruit("Hạt dẻ", 3.0)
    elapsed = time.time() - t0

    assert elapsed < 8.0, f"Must abort within reasonable time, took {elapsed:.2f}s"
    assert engine.state == FarmState.STOPPED
    assert any("QUÁ THỜI GIAN" in l for l in logs)
    assert any("HƯỚNG DẪN KHẮC PHỤC" in l for l in logs)

def test_step1_timeout_watchdog_in_fast_harvest():
    engine = FarmEngine(capturer=BlankCapturer())
    engine.step1_timeout_s = 0.5
    logs = []
    engine.log_callback = logs.append

    t0 = time.time()
    engine._harvest_multi_fruits_fast(["Xoài", "Dừa"])
    elapsed = time.time() - t0

    assert elapsed < 8.0, f"Must abort within reasonable time, took {elapsed:.2f}s"
    assert engine.state == FarmState.STOPPED
    assert any("QUÁ THỜI GIAN" in l for l in logs)
    assert any("HƯỚNG DẪN KHẮC PHỤC" in l for l in logs)
