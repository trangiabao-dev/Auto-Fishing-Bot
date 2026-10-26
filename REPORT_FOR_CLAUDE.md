# BÁO CÁO KỸ THUẬT & NHẬT KÝ KIẾN TRÚC HỆ THỐNG
**Dành cho: Senior Reviewer / Claude / Nhà phát triển hệ thống**  
**Dự án**: Auto-Fishing Bot - Play Together (Vision-Based Closed-Loop Automation)  
**Mã nguồn**: `D:\ToolPT` (Git Repository: `D:\GitHub Desktop\GitHub\Auto-Fishing-Bot`)  
**Ngày cập nhật**: 26/10/2026  

---

## 1. TỔNG QUAN KIẾN TRÚC & TIẾN TRÌNH TỪ HÔM QUA ĐẾN NAY

Dự án đã trải qua quá trình tái cấu trúc từ một hệ thống **Open-Loop State Machine** (chạy mù theo bước thời gian cứng) sang kiến trúc **Closed-Loop Perception-Driven Engine** (ra quyết định hoàn toàn dựa trên cảm biến thị giác và vòng lặp khép kín tự phục hồi).

### Bản đồ Module Cốt Lõi (`core/` & Engine):
1. [bot_engine.py](file:///d:/ToolPT/bot_engine.py): Điều phối luồng trạng thái (`BotEngine`), quản lý chu kỳ sống (`STAGE_1`, `STAGE_2`, `STAGE_3`, `STAGE_REPAIR`).
2. [app_gui.py](file:///d:/ToolPT/app_gui.py): Giao diện người dùng thế hệ mới (Catppuccin Mocha Pro, 2 Tab phân tách, Chế độ Mini Bar 145px, Độ mờ xuyên thấu 88%).
3. [core/scale_aware.py](file:///d:/ToolPT/core/scale_aware.py): Ngân hàng template co giãn theo camera zoom (`BiteTemplateBank`), tính toán ROI động (`scaled_head_roi`), tích hợp Dynamic Range Gate chặn nhiễu phẳng.
4. [core/static_veto.py](file:///d:/ToolPT/core/static_veto.py): Bộ lọc vật tĩnh trôi dạt (`StaticVeto`), so sánh dấu `!` với lịch sử $0.8 \div 3.0$s trước để loại bỏ mỏm đá, thuyền, bọt nước cố định.
5. [core/shape_veto.py](file:///d:/ToolPT/core/shape_veto.py): Bộ lọc hình dạng (`is_cloud_like`), nhận diện đám mây bong bóng suy nghĩ/nhiệm vụ của NPC để triệt tiêu kích hoạt giả.
6. [core/plate_locator.py](file:///d:/ToolPT/core/plate_locator.py): Bộ định vị tên người chơi (`PlateLocator` & `PlateTracker`) dựa trên mặt nạ màu Neon-Green (HSV), tìm blob connected components và bám vết nhân vật thời gian thực.
7. [core/region_scan.py](file:///d:/ToolPT/core/region_scan.py): Quét siêu tốc các nút cố định (`RegionScanner`, single-scale, ROI hẹp, < 25ms/frame).
8. [core/progress_watchdog.py](file:///d:/ToolPT/core/progress_watchdog.py): Giám sát tiến độ, tự động lưu ảnh bằng chứng vào `corpus/stuck/` nếu bot bị kẹt trạng thái $> 30$s.

---

## 2. BỐN VẤN ĐỀ LỚN ĐÃ CHẨN ĐOÁN & XỬ LÝ TẬN GỐC

### VẤN ĐỀ 1: Bot Ấn Nhầm Nút "Khoe" Kế Bên Thay Vì "Bảo Quản" (STAGE 3)

- **Triệu chứng**: Khi câu được cá, bot ấn nhầm nút "Khoe" ở bên trái thay vì nút "Bảo Quản", mở ra màn hình chụp ảnh/chia sẻ mạng xã hội của game. Sau 30s không thể tiếp tục, watchdog báo động `⚠️ [KẸT >30s]`.
- **Nguyên nhân gốc (Root Cause)**:
  1. *Lỗi miền không gian (Spatial Flaw)*: Khai báo cũ `ScanTarget("keep", ..., (0.45, 0.45, 1.00, 1.00))` quét từ $X \ge 864$. Trên chuẩn màn hình 1080p, nút "Khoe" nằm ở $X \in [950, 1220]$, còn "Bảo Quản" ở $X \in [1350, 1650]$. Vùng quét cũ đã **ôm trọn cả 2 nút**.
  2. *Nhiễu Template*: Trong `assets/B3_BaoQuan/` tồn tại file cũ `nut_bao_quan.png` (126x55 mờ, chỉ chứa viền xanh viên thuốc không rõ chữ). Vì 2 nút có cùng màu xanh dương dạng capsule, thuật toán đã khớp nhầm sang nút bên trái.
- **Giải pháp Senior đã triển khai**:
  - **Spatial Isolation (Cô lập hình học)**: Thu hẹp vùng quét mục tiêu `keep` thành `(0.65, 0.65, 0.95, 0.95)` trong [bot_engine.py](file:///d:/ToolPT/bot_engine.py#L90), tương đương $X \in [1248, 1824]$. Vùng $X < 1248$ (nơi chứa nút "Khoe") bị cắt bỏ hoàn toàn trước khi xử lý ảnh.
  - **Template Sanitization**: Chuyển file mờ `nut_bao_quan.png` vào `assets/_backup_old/B3_BaoQuan/`, chỉ giữ lại duy nhất bản chuẩn 1080p có chữ sắc nét `baoquan_1080p.jpg`.
  - **Kiểm chứng**: Đã thêm unit test `test_keep_region_excludes_share_button` trong [tests/test_region_scan.py](file:///d:/ToolPT/tests/test_region_scan.py).

---

### VẤN ĐỀ 2: Cửa Sổ Ứng Dụng Quá Dài, Che Tầm Nhìn Game & Log Rối Mắt

- **Triệu chứng**: Cửa sổ cũ kích thước $620 \times 720$ chiếm 70% chiều cao màn hình, đè lên phím điều khiển câu cá và mặt hồ; log hiển thị lẫn lộn các thông số benchmark kỹ thuật (`⏱ [ĐO ĐẠC]`, `gap p50/p95`) gây ngợp.
- **Giải pháp Senior đã triển khai trong [app_gui.py](file:///d:/ToolPT/app_gui.py)**:
  1. **Thiết kế Compact Default**: Giảm kích thước xuống $460 \times 410$ (giảm gần một nửa chiều cao), tinh gọn toàn bộ header và card trạng thái lên 1 dòng.
  2. **Chế độ Mini Bar Siêu Nhỏ (Nút `🗕 Thu Gọn`)**: Khi bấm thu gọn, toàn bộ khung log ẩn đi, cửa sổ co lại chỉ còn **145px** (chiếm ~13% màn hình). Tích hợp **Live Ticker** hiển thị tức thì hành động bot đang làm.
  3. **Tùy chọn `🌓 Mờ nhẹ` (Transparency 88%)**: Cho phép cửa sổ bán trong suốt để nhìn xuyên qua thấy khung cảnh game phía sau.
  4. **Phân tách 2 Tab độc lập**:
     - *Tab 1: 📋 Trạng Thái Bot (Rút Gọn)*: Lọc sạch telemetry rác, hiển thị quy trình người dùng dễ hiểu, có đường kẻ phân cách màu cam nổi bật `━━━━━━━━━━━━━━━━━━` sau mỗi vòng câu thành công.
     - *Tab 2: 🛠 Log Chi Tiết (Debug)*: Giữ nguyên 100% telemetry, timestamp, có nút **`📋 Copy Toàn Bộ Log`** xuất clipboard 1-click.

---

### VẤN ĐỀ 3: Làn Hơi Nước/Sương Mù Di Chuyển Tại Thác Nước & Nhiễu Bão Gây Giật Hụt Liên Tục

- **Triệu chứng**: Khi câu cá tại khu vực Thác Nước hoặc khi trời giông bão, bot liên tục giật cần hụt sau $2 \div 3$ giây quăng cần, log báo:
  `⚡ [BƯỚC 2] CÁ CẮN! Dấu '!' tại (507, 249) [chamthan_vang.png] (0.775)`
- **Bóc tách nguyên nhân từ dữ liệu `corpus/missed/`**:
  1. *Làn khói/hơi nước di chuyển liên tục*: `StaticVeto` chỉ bắt được vật đứng yên ở cùng pixel từ 0.8s trước. Hơi nước thác nước cuộn trào và bay đổi hình dạng mỗi 100ms, nên `StaticVeto` thấy mỗi frame là vật thể mới và không chặn.
  2. *Lỗi kinh điển Low-Variance Inflation Anomaly*:
     Công thức `TM_CCOEFF_NORMED` luôn chia cho độ lệch chuẩn $\sigma$ của vùng ảnh:
     $$\text{Score} = \frac{\sum (I - \bar{I})(T - \bar{T})}{\sqrt{\sum (I - \bar{I})^2 \sum (T - \bar{T})^2}}$$
     Làn khói mờ ảo trên nền đất/nước tối ban đêm có độ tương phản cực kỳ phẳng: mức xám chỉ dao động từ **$73 \to 78$** ($\Delta = 5$ mức xám, độ lệch chuẩn $\sigma = 0.81$).
     Vì mẫu số $\sqrt{\sum (I - \bar{I})^2}$ quá nhỏ, thuật toán **tự động phóng đại viền mờ của làn khói lên hàng chục lần**, biến một gợn khói mờ xỉn thành dấu `!` giả có điểm tới **0.775**!
     Trong khi cá cắn thật: Dấu `!` phát sáng chói lọi có $\Delta \ge 150$, $\sigma \ge 46.0$, độ sáng $V_{max} \ge 240$.
- **Giải pháp Senior đã triển khai (Triple Guardrails)**:
  - **Guardrail A (Dynamic Range Gate)** trong [core/scale_aware.py](file:///d:/ToolPT/core/scale_aware.py#L90-L105): Kiểm tra vùng ảnh $\Delta = I_{max} - I_{min}$. Nếu $\Delta < 14$, lập tức loại bỏ. Triệt tiêu hoàn toàn nhiễu phẳng từ gốc.
  - **Guardrail B (Luminance Gate)** trong [bot_engine.py](file:///d:/ToolPT/bot_engine.py#L610-L625): Dấu `!` luôn là icon phát quang ban đêm ($V_{max} \ge 175$). Bất kỳ vùng nào có $V_{max} < 135$ (như khói mờ, bóng bùn) bị hủy bỏ.
  - **Guardrail C (Relocation Grace Period)** trong [bot_engine.py](file:///d:/ToolPT/bot_engine.py#L530-L540): Khi vùng đầu di chuyển $> 20$px do xoay camera, bot chờ $0.6$s để `StaticVeto` tích lũy lịch sử phông nền mới.
  - **Kiểm chứng**: 6/6 clip giật hụt ở thác nước đã bị chặn 100%, 117/117 unit tests pass.

---

### VẤN ĐỀ 4 (Hiện Tại - Log 10:07): Nhiễu Đám Đông (Crowd Occlusion) Khi Người Chơi Khác Đứng Đè Lên

- **Phân tích log mới nhất của bạn (10:07:02 - 10:07:08)**:
  ```text
  [10:07:02] -> Định vị nhân vật 'columbinaa.png' (scale 1.11): vùng đầu = (825, 300, 271, 211)
  [10:07:04] -> Định vị nhân vật 'columbinaa.png' (scale 1.29): vùng đầu = (1139, 188, 314, 244)
  [10:07:06] -> PHÁT HIỆN DẤU CHẤM THAN TRÊN ĐẦU! (0.78 [chamthan_vang.png] tại 1260, 389)
  [10:07:06] -> KÍCH HOẠT: Giật cần câu tại nút bấm (1741, 990)
  [10:07:08] 🔄 [TỰ SỬA TRẠNG THÁI] Nút Giật Cần vẫn còn (0.75)! Màn hình chứng minh vẫn đang câu cá...
  ```
- **Hiện tượng thực tế từ ảnh chụp `media_1792983994423_d24ced1c.png`**:
  - Tại vị trí câu cá có **rất nhiều người chơi khác đứng tụ tập đè lên nhau**:
    - Nhân vật chính `columbinaa` có bảng danh hiệu màu hồng: *"Ngồi không ăn lương"*.
    - Ngay sát bên cạnh và phía sau có các người chơi khác (`nqwy`, `xiqi`) **cũng đeo danh hiệu y hệt** *"Ngồi không ăn lương"* và icon quả táo xanh.
    - Phía trên đầu đám đông có **vương miện vàng (👑)** và **hai tay cầm chơi game (🎮🎮)** lơ lửng.
  - **Hệ quả**:
    1. Lúc 10:07:02, bot khóa đúng vị trí của bạn tại `(825, 300)`.
    2. Đến 10:07:04, `PlateTracker` bị nhảy mục tiêu (Target Hijack) sang một người chơi khác trong đám đông ở tọa độ `(1139, 188)`.
    3. Tại `(1260, 389)` (vùng danh hiệu / phụ kiện phát sáng của người chơi kế bên), bot bị khớp nhầm với `chamthan_vang.png`.
    4. Cú giật cần non khiến cần câu bị rút lên, vòng câu bị hủy.

---

## 3. TRẠNG THÁI MÃ NGUỒN TRÊN GITHUB & HƯỚNG DẪN PUSH

Kho mã nguồn cục bộ `D:\GitHub Desktop\GitHub\Auto-Fishing-Bot` hiện đã commit sạch sẽ toàn bộ các thay đổi trên:
1. `3f5acc8`: `feat(gui): overhaul modern 2-tab UI and isolate keep button from share button`
2. `99e494c`: `feat(gui): make UI compact and add Mini mode to avoid blocking game view`
3. `b618c43`: `fix(vision): eliminate waterfall mist and dark flat noise false positives via dynamic range and luminance gates`

> [!IMPORTANT]
> **Thao tác Push lên GitHub**:
> Vì Git trên máy tính sử dụng Windows Credential Manager / GitHub Desktop để xác thực tài khoản, bạn chỉ cần mở ứng dụng **GitHub Desktop** trên màn hình và bấm nút **"Push origin"** (hoặc mở terminal gõ `git push`). Toàn bộ 3 commit mới nhất sẽ lập tức có mặt trên repo GitHub của bạn để Claude hoặc bất kỳ ai có thể đọc trực tiếp!
