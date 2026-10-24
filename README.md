# 🎣 AUTO FISHING TOOL (PLAY TOGETHER)
> Tool tự động câu cá bằng thị giác máy tính (Computer Vision), hoạt động 100% offline trên máy cá nhân, không can thiệp bộ nhớ game.

---

## 🌟 Tính Năng Nổi Bật
- **Nhận diện siêu tốc**: Tự động phát hiện phao câu quăng cần (B1), canh dấu chấm than trên đỉnh đầu nhân vật (B2), và tự lưu cá vào balo (B3).
- **Tự bám theo nhân vật (Dynamic Head ROI)**: Lọc màu chữ xanh neon của tên nhân vật để bám theo đỉnh đầu thời gian thực, triệt tiêu 100% việc giật nhầm do cá bơi hay bóng nước.
- **Phản xạ 0ms khi cá cắn**: Nút giật cần được định vị sẵn từ lúc chờ, click ngay lập tức khi dấu chấm than vừa lóe lên.
- **Chống khóa tài khoản (Human-like Randomization)**: Tọa độ click chuột ngẫu nhiên trong vùng nút và thời gian nhấn giữ chuột ngẫu nhiên (40ms - 90ms).
- **Giao diện hiện đại**: Nền tối Dark Mode, đếm số cá câu được, bảng log trực quan, luôn nổi trên cửa sổ game.

---

## 🚀 Hướng Dẫn Sử Dụng Nhanh

### 1. Trên máy của bạn:
- Mở shortcut **`Auto Fishing Tool`** ngoài Desktop (hoặc nhấp đúp file `ChayTool.bat`).
- Vào game, cầm cần câu đứng sát mép nước.
- Nhấn **`[F1]`** trên bàn phím: Bot sẽ tự quăng cần và câu liên tục.
- Nhấn **`[F2]`** bất kỳ lúc nào để dừng lại.

---

## 👥 Hướng Dẫn Chia Sẻ Cho Bạn Bè

Để gửi tool này cho bạn bè sử dụng trên máy của họ:

### Bước 1: Nén thư mục gửi cho bạn bè
- Nén toàn bộ thư mục `ToolPT` thành file zip (ví dụ `AutoFishing_PlayTogether.zip`) rồi gửi qua Zalo/Drive/USB.

### Bước 2: Bạn bè cài đặt trên máy họ
1. Giải nén thư mục ra ổ cứng (ví dụ `D:\ToolPT` hoặc `C:\ToolPT`).
2. Yêu cầu: Máy cần có sẵn **Python 3.10+** (nếu chưa có thì tải tại https://www.python.org/ - **nhớ tick vào ô "Add Python to PATH"** khi cài).
3. Nhấp đúp vào file **`CaiDat.bat`** để tool tự động cài các thư viện cần thiết (`opencv`, `mss`, `pydirectinput`, `keyboard`, `pillow`).
4. Nhấp đúp vào **`ChayTool.bat`** để mở ứng dụng!

### Bước 3: Cấu hình Tên Nhân Vật cho bạn bè
Vì tên trong game của bạn bè khác tên của bạn (`mumyua`), bạn bè chỉ cần:
1. Chụp ảnh bảng tên nhân vật màu xanh lá của họ trong game.
2. Lưu đè vào file `assets/B2_DangCau/vat_neo_ten/mumyua.png` (hoặc đổi đường dẫn trong file `config.json` ở dòng `"b2_ten_nhan_vat"`).

---

## 🗺️ Lộ Trình Nâng Cấp Tương Lai (Roadmap)

Thiết kế kiến trúc State Machine trong `bot_engine.py` tuân theo nguyên lý **Open-Closed Principle (SOLID)**, sẵn sàng mở rộng các tính năng tiếp theo:

1. **🚨 Tự động báo Đầy Túi (Inventory Full Alert)**:
   - Phát hiện popup thông báo đầy túi đồ khi câu cá.
   - Phát âm thanh chuông cảnh báo (`Beep`) để người chơi biết và tự động dừng bot an toàn.
   - *(Nâng cao)*: Tự mở balo ➔ chọn Bán Cá / Tách Cá nhanh rồi câu tiếp.

2. **🔧 Tự động Sửa Cần Câu khi hỏng (Broken Rod Auto-Repair)**:
   - Khi độ bền cần câu về 0, game hiện thông báo sửa cần câu bằng tiền sao.
   - Bot nhận diện popup ➔ tự động click nút "Sửa" và "Xác nhận" ➔ tiếp tục chu trình quăng cần tự động 100% không cần người ngồi canh.
