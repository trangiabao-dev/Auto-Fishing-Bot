import os
import time
import numpy as np
import pytest
from core.missed_catch_logger import MissedCatchLogger


def test_missed_catch_logger_creates_folders_and_paired_files(tmp_path):
    base_dir = str(tmp_path / "test_debug_hut_ca")
    logger = MissedCatchLogger(base_dir=base_dir)

    assert os.path.isdir(os.path.join(base_dir, "anh"))
    assert os.path.isdir(os.path.join(base_dir, "debug"))

    full_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    head_frame = np.full((150, 150, 3), 128, dtype=np.uint8)
    head_coords = (500, 200, 150, 150)
    plate_info = {
        "state": "TRACKING",
        "name": "columbinaa.png",
        "score": 0.85,
        "plate": (520, 350, 120, 35),
        "scale": 1.2
    }
    bite_analysis = {
        "threshold": 0.80,
        "top_score": 0.74,
        "top_template": "mautim_oaihuong.png",
        "matches": {
            "mautim_oaihuong.png": 0.74,
            "chamthan_vang.png": 0.62
        }
    }

    case_id = logger.log_incident(
        full_frame=full_frame,
        head_roi_frame=head_frame,
        head_roi_coords=head_coords,
        reason="Cần câu bị giật lên lại / chưa vào câu (mất cá)",
        plate_info=plate_info,
        bite_analysis=bite_analysis,
        duration_s=25.4,
        scan_count=350,
    )

    assert case_id.startswith("hut_ca_")

    # Chờ worker thread ghi xong
    time.sleep(0.3)

    img_path = os.path.join(base_dir, "anh", f"{case_id}_man_hinh.png")
    head_img_path = os.path.join(base_dir, "anh", f"{case_id}_vung_dau.png")
    debug_path = os.path.join(base_dir, "debug", f"{case_id}_debug.txt")

    assert os.path.exists(img_path), f"Missing {img_path}"
    assert os.path.exists(head_img_path), f"Missing {head_img_path}"
    assert os.path.exists(debug_path), f"Missing {debug_path}"

    with open(debug_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert case_id in content
    assert "mautim_oaihuong.png" in content
    assert "0.74" in content
    assert "columbinaa.png" in content
