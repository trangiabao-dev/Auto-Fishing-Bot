# BỘ DỮ LIỆU THỰC NGHIỆM THỊ GIÁC & TELEMETRY (PLAY TOGETHER AUTO FISHING)

Bộ dữ liệu này gồm 15 ảnh chụp màn hình thực tế và file telemetry.jsonl ghi lại từ 3 khu vực:

### 1. Hồ Halloween (Nghẽn luồng quét / Scan Freeze):
- Halloween_01_GocCheo.png: Góc chéo (Bên phải), locate ngốn 1005ms, gap max 1220ms (P_miss 53%).
- Halloween_02_TopDown.png: Góc nhìn trên cao, locate ngốn 1579ms, gap max 1646ms (đứt dây hụt cá).
- Halloween_03_GocNgang.png: Góc ngang, nhân vật đứng lệch trái (X=651).
- Halloween_04_ChamThanXanhLa.png: Dấu chấm than màu xanh lá cây đặc trưng tại sự kiện Halloween.

### 2. Biển Sâu (Báo động giả tĩnh & Vòng lặp vô tận):
- BienSau_01_Phai_Ngay_KetVongLap2.4s.png: Góc phải (buổi sáng), chi tiết mạ vàng của thuyền khớp 0.83 với chamthan_vang.png tại (623, 229), gây kẹt vòng lặp giật hụt đúng 2.4s sau khi quăng cần.
- BienSau_02_TopDown_Dem_Dinh100.png: Góc nhìn trên đỉnh đầu xuống (ban đêm), dính cá 100% (2/2 con).
- BienSau_03_Trai_Dem_Dinh100.png: Góc bên trái nhân vật (ban đêm), dính cá 100% (2/2 con, chấm than hồng 0.92-0.96).
- BienSau_04_Phai_Dem_Dinh50.png: Góc bên phải (ban đêm), vật thể tĩnh dao động quanh 0.73 -> 2 lần giật non hụt, 2 lần dính cá thật.

### 3. Thác Nước (Báo động giả động do môi trường & Lỗi Zoom cận cảnh):
- ThacNuoc_01_Phai_Dung.png: Góc phải chuẩn bị câu, người chơi BT21 và pet đứng xung quanh.
- ThacNuoc_02_Phai_ZoomCanCanh_MatDau.png: Zoom cận cảnh nhân vật -> bảng tên to tỉ lệ 2.0x -> NameLocator bị mất dấu (loc = None).
- ThacNuoc_03_TopDown_Dung.png: Góc top-down đứng trước khi câu.
- ThacNuoc_04_TopDown_Ngoi.png: Góc top-down ngồi câu, bọt nước và sóng quanh phao khớp 0.73 - 0.82 (giật non sau 2.9s - 3.2s).
- ThacNuoc_05_Trai_Ngoi.png: Góc trái ngồi câu, bọt nước thác nước khớp 0.76 (giật non sau 3.1s - 4.3s).
- ThacNuoc_06_GocNgang_Cu.png & ThacNuoc_07_TopDown_Cu.png: Các ảnh tham chiếu từ phiên trước.
