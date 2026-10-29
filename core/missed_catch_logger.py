"""
Chuyên biệt ghi vết các sự cố hụt cá (Missed Catch Incident Logger).

Tạo thư mục `debug_hut_ca` với 2 thư mục con rõ ràng:
  - `debug_hut_ca/anh/`: Lưu ảnh toàn màn hình (có khoanh vùng đầu & bảng tên) và ảnh cận cảnh vùng đầu.
  - `debug_hut_ca/debug/`: Lưu file text báo cáo nguyên nhân kỹ thuật chi tiết tương ứng với từng ảnh.

Tên file được liên kết chặt chẽ qua mã sự cố: `hut_ca_YYYYMMDD_HHMMSS`.
Ghi trên luồng nền (background thread), không gây lag hay chặn vòng lặp chính của bot.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class MissedCatchLogger:
    def __init__(self, base_dir: str = "debug_hut_ca"):
        self.base_dir = os.path.abspath(base_dir)
        self.img_dir = os.path.join(self.base_dir, "anh")
        self.debug_dir = os.path.join(self.base_dir, "debug")
        
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

    def log_incident(
        self,
        full_frame: Optional[np.ndarray],
        head_roi_frame: Optional[np.ndarray],
        head_roi_coords: Optional[Tuple[int, int, int, int]],
        reason: str,
        plate_info: Optional[Dict[str, Any]] = None,
        bite_analysis: Optional[Dict[str, Any]] = None,
        duration_s: float = 0.0,
        scan_count: int = 0,
    ) -> str:
        """
        Ghi nhận sự cố hụt cá bất đồng bộ trên luồng phụ.
        Trả về mã sự cố: hut_ca_YYYYMMDD_HHMMSS.
        """
        now = datetime.now()
        case_id = f"hut_ca_{now.strftime('%Y%m%d_%H%M%S')}"

        # Copy mảng ảnh để tránh race-condition với luồng chụp màn hình
        full_copy = full_frame.copy() if full_frame is not None else None
        head_copy = head_roi_frame.copy() if head_roi_frame is not None else None

        worker = threading.Thread(
            target=self._write_files,
            args=(case_id, now, full_copy, head_copy, head_roi_coords, reason,
                  plate_info, bite_analysis, duration_s, scan_count),
            daemon=True,
            name=f"missed-logger-{case_id}"
        )
        worker.start()
        return case_id

    def _write_files(
        self,
        case_id: str,
        now: datetime,
        full_frame: Optional[np.ndarray],
        head_roi_frame: Optional[np.ndarray],
        head_roi_coords: Optional[Tuple[int, int, int, int]],
        reason: str,
        plate_info: Optional[Dict[str, Any]],
        bite_analysis: Optional[Dict[str, Any]],
        duration_s: float,
        scan_count: int,
    ) -> None:
        try:
            # 1. Lưu ảnh toàn màn hình có chú thích trực quan
            full_img_name = f"{case_id}_man_hinh.png"
            full_img_path = os.path.join(self.img_dir, full_img_name)
            
            if full_frame is not None:
                annotated = full_frame.copy()
                
                # Lồng ảnh vùng đầu lúc đạt đỉnh (Peak Candidate Frame) vào ảnh toàn cảnh
                if head_roi_coords is not None and head_roi_frame is not None and head_roi_frame.size > 0:
                    hx, hy, hw, hh = head_roi_coords
                    fh, fw = annotated.shape[:2]
                    # Clamp to frame dimensions
                    cl_x0, cl_y0 = max(0, hx), max(0, hy)
                    cl_x1, cl_y1 = min(fw, hx + hw), min(fh, hy + hh)
                    if cl_x1 > cl_x0 and cl_y1 > cl_y0:
                        try:
                            resized_head = cv2.resize(head_roi_frame, (cl_x1 - cl_x0, cl_y1 - cl_y0))
                            annotated[cl_y0:cl_y1, cl_x0:cl_x1] = resized_head
                        except Exception:
                            pass

                # Khoanh vùng đầu đang quét (màu đỏ)
                if head_roi_coords is not None:
                    hx, hy, hw, hh = head_roi_coords
                    cv2.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 2)
                    p_score = bite_analysis.get('top_score', 0.0) if bite_analysis else 0.0
                    p_time = bite_analysis.get('peak_time_s', 0.0) if bite_analysis else 0.0
                    cv2.putText(annotated, f"Peak Candidate ({p_score:.2f} at +{p_time:.1f}s)", (hx, max(20, hy - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                
                # Khoanh bảng tên nhân vật nếu có (màu xanh lá)
                if plate_info and "plate" in plate_info and plate_info["plate"] is not None:
                    p = plate_info["plate"]
                    px, py, pw, ph = p[0], p[1], p[2], p[3]
                    cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
                    cv2.putText(annotated, f"Plate: {plate_info.get('score', 0):.2f}", (px, max(20, py - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imwrite(full_img_path, annotated)

            # 2. Lưu ảnh cận cảnh vùng đầu
            head_img_name = f"{case_id}_vung_dau.png"
            head_img_path = os.path.join(self.img_dir, head_img_name)
            if head_roi_frame is not None and head_roi_frame.size > 0:
                cv2.imwrite(head_img_path, head_roi_frame)

            # 3. Tạo file báo cáo debug chi tiết
            report_name = f"{case_id}_debug.txt"
            report_path = os.path.join(self.debug_dir, report_name)
            
            p_info = plate_info or {}
            b_info = bite_analysis or {}

            lines = [
                "=" * 72,
                f"BÁO CÁO SỰ CỐ HỤT CÁ - MÃ: {case_id}",
                "=" * 72,
                f"Thời gian ghi nhận: {now.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Lý do sự cố       : {reason}",
                f"Thời gian câu     : {duration_s:.1f}s (đã quét {scan_count} lần)",
                "",
                "--- [1] THÔNG TIN ĐỊNH VỊ NHÂN VẬT (PLATE TRACKER) ---",
                f"Trạng thái theo dõi : {p_info.get('state', 'UNKNOWN')}",
                f"Tên template        : {p_info.get('name', 'N/A')}",
                f"Độ khớp bảng tên    : {p_info.get('score', 0.0):.3f}",
                f"Vị trí bảng tên     : {p_info.get('plate', 'N/A')}",
                f"Tỉ lệ zoom (Scale)  : {p_info.get('scale', 1.0):.2f}x",
                "",
                "--- [2] VÙNG QUÉT ĐẦU (HEAD ROI) ---",
                f"Tọa độ quét trên màn hình: {head_roi_coords or 'Chưa xác định'}",
                f"Đánh giá trực quan       : Đã xuất ảnh vùng đầu '{head_img_name}' để kiểm chứng.",
                "",
                "--- [3] PHÂN TÍCH SO KHỚP DẤU CHẤM THAN '!' (PEAK CANDIDATE MEMORY) ---",
                f"Ngưỡng yêu cầu tối thiểu : >= {b_info.get('threshold', 0.80):.2f}",
                f"Điểm cao nhất đạt được   : {b_info.get('top_score', 0.0):.3f} (Template: {b_info.get('top_template', 'Không có')})",
                f"Thời điểm đạt đỉnh       : +{b_info.get('peak_time_s', 0.0):.1f}s sau khi thả mồi",
                f"Nguyên nhân bot không giật: {b_info.get('peak_reason', 'Không đạt ngưỡng kích hoạt')}",
                f"Chi tiết các mẫu thử     :",
            ]

            matches = b_info.get("matches", {})
            if matches:
                for tpl_name, sc in sorted(matches.items(), key=lambda x: x[1], reverse=True)[:5]:
                    lines.append(f"  * {tpl_name:24s}: {sc:.3f}")
            else:
                lines.append("  (Không có mẫu nào đạt mức tối thiểu hoặc không ghi nhận được)")

            lines.extend([
                "",
                "--- [4] DANH SÁCH FILE LIÊN QUAN TRONG THƯ MỤC 'debug_hut_ca' ---",
                f"1. Ảnh toàn màn hình: anh/{full_img_name}",
                f"2. Ảnh cận cảnh đầu : anh/{head_img_name}",
                f"3. Báo cáo debug    : debug/{report_name}",
                "=" * 72,
                ""
            ])

            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        except Exception as e:
            # Ghi lỗi âm thầm, tuyệt đối không làm sập tiến trình bot
            print(f"[MissedCatchLogger ERROR] {e}")
