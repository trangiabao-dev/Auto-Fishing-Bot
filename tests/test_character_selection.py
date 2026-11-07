import os
import cv2
import numpy as np
import pytest

from core.plate_locator import PlateLocator
from app_gui import get_available_characters, save_config, load_config


def test_get_available_characters_filters_subdirs_and_badges(tmp_path):
    archive_dir = str(tmp_path / "archive")
    os.makedirs(archive_dir)
    # Valid characters
    (tmp_path / "archive" / "columbinaa.png").write_text("fake")
    (tmp_path / "archive" / "teffiny.png").write_text("fake")
    # Subdir (e.g. badges/)
    badges_dir = os.path.join(archive_dir, "badges")
    os.makedirs(badges_dir)
    (tmp_path / "archive" / "badges" / "anchor_cudanmoiden.png").write_text("fake")
    # Non-image file
    (tmp_path / "archive" / "notes.txt").write_text("fake")

    chars = get_available_characters(archive_dir)
    filenames = [f for d, f in chars]
    assert "columbinaa.png" in filenames
    assert "teffiny.png" in filenames
    assert "anchor_cudanmoiden.png" not in filenames
    assert "notes.txt" not in filenames


def test_plate_locator_single_template_isolation():
    col_path = r"D:\ToolPT\assets\B2_DangCau\vat_neo_ten\archive\columbinaa.png"
    tef_path = r"D:\ToolPT\assets\B2_DangCau\vat_neo_ten\archive\teffiny.png"
    if not (os.path.exists(col_path) and os.path.exists(tef_path)):
        pytest.skip("Template files not present")

    img_col = cv2.imread(col_path)
    img_tef = cv2.imread(tef_path)

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[400:400+img_tef.shape[0], 500:500+img_tef.shape[1]] = img_tef
    frame[400:400+img_col.shape[0], 900:900+img_col.shape[1]] = img_col

    # 1. Locator for Columbinaa only
    loc_col = PlateLocator(col_path)
    cands_col = loc_col.candidates(frame)
    assert len(cands_col) == 1
    assert cands_col[0].name == "columbinaa.png"
    assert cands_col[0].plate.x == 900

    # 2. Locator for Teffiny only
    loc_tef = PlateLocator(tef_path)
    cands_tef = loc_tef.candidates(frame)
    assert len(cands_tef) == 1
    assert cands_tef[0].name == "teffiny.png"
    assert cands_tef[0].plate.x == 500


def test_config_character_lock_persistence(tmp_path):
    cfg_file = str(tmp_path / "config.json")
    data = {
        "fishing": {
            "selected_character": "teffiny.png",
            "character_locked": True
        }
    }
    save_config(data, cfg_file)
    loaded = load_config(cfg_file)
    assert loaded["fishing"]["selected_character"] == "teffiny.png"
    assert loaded["fishing"]["character_locked"] is True
