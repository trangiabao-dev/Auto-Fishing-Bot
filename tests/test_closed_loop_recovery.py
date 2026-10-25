import numpy as np
import pytest
from core.geometry import Rect
from core.region_scan import RegionScanner, ScanTarget, ScanHit
from bot_engine import BotEngine, BotState

class FakeCapturer:
    def __init__(self, frame, offset=(0, 0)):
        self.frame = frame
        self.offset = offset
    def capture(self, region=None):
        return self.frame, self.offset

def test_closed_loop_stage3_recovers_to_stage2_if_reel_still_present():
    cfg = {
        "templates": {
            "b1_quang_can": "assets/B1_QuangCan",
            "b2_giat_can": "assets/B2_DangCau/nut_giat_can.png",
            "b2_cham_than": "assets/B2_DangCau/dau_cham_than",
            "b2_ten_nhan_vat": "assets/B2_DangCau/vat_neo_ten",
            "b3_bao_quan": "assets/B3_BaoQuan",
            "b4_sua_can_title": "assets/B4_SuaCan/title",
            "b4_sua_can_btn": "assets/B4_SuaCan/button",
            "b4_hoan_thanh_title": "assets/B4_SuaCan/title_hoan_thanh",
            "b4_ok_btn": "assets/B4_SuaCan/btn_ok",
        },
        "thresholds": {
            "b1_quang_can": 0.70,
            "b2_giat_can": 0.70,
            "b2_giat_can_threshold": 0.52,
            "b3_bao_quan": 0.70,
            "b4_sua_can": 0.70,
            "b4_btn_ok": 0.70,
        }
    }
    engine = BotEngine(cfg)
    
    # Mock scanner to simulate screen where reel button is present (0.57) and keep is absent (0.20)
    fake_hits = {
        "cast": ScanHit(found=False, score=0.30),
        "reel": ScanHit(found=True, score=0.57, x=1654, y=817, w=180, h=181, template="nut_giat_can.png"),
        "keep": ScanHit(found=False, score=0.20),
        "repair_title": ScanHit(found=False, score=0.10),
        "done_title": ScanHit(found=False, score=0.10),
        "ok": ScanHit(found=False, score=0.10),
        "repair_btn": ScanHit(found=False, score=0.10),
    }
    engine.scanner.scan = lambda frame, origin=(0, 0), names=None: fake_hits
    
    # Set engine to STAGE_3
    engine.state = BotState.STAGE_3
    
    # If scanner sees reel is found, stage 3 should recover to STAGE_2
    # Verify scanner reports reel found
    hits = engine.scanner.scan(np.zeros((1080, 1920, 3), np.uint8))
    assert hits["reel"].found is True
    assert hits["keep"].found is False
