import os
import sys
import time
import ctypes
import cv2
import numpy as np
import mss
import keyboard

# Ensure DPI awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_mouse_pos():
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def real_time_inspector():
    print("=" * 60)
    print("  CHẾ ĐỘ 1: THEO DÕI TỌA ĐỘ CHUỘT THỜI GIAN THỰC")
    print("  Rê chuột đến vị trí trên màn hình để xem tọa độ (X, Y).")
    print("  Nhấn tổ hợp phím Ctrl + C để thoát về menu.")
    print("=" * 60)
    try:
        while True:
            x, y = get_mouse_pos()
            sys.stdout.write(f"\r >> Tọa độ chuột hiện tại: X = {x:4d} | Y = {y:4d}    ")
            sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nĐã dừng chế độ theo dõi.\n")

def snip_and_save():
    print("=" * 60)
    print("  CHẾ ĐỘ 2: CÔNG CỤ CHỤP ẢNH MẪU & LẤY TỌA ĐỘ VÙNG (ROI)")
    print("  1. Rê chuột vào GÓC TRÊN - BÊN TRÁI của hình ảnh, nhấn phím [F7]")
    print("  2. Rê chuột vào GÓC DƯỚI - BÊN PHẢI của hình ảnh, nhấn phím [F8]")
    print("=" * 60)

    p1 = None
    p2 = None

    print("\n👉 Đang chờ bạn nhấn [F7] để lấy điểm 1 (Góc trên-trái)...")
    while True:
        if keyboard.is_pressed('f7'):
            p1 = get_mouse_pos()
            print(f" ✔ Điểm 1 (Góc trên-trái): {p1}")
            time.sleep(0.5)  # Tránh bấm dính
            break
        time.sleep(0.02)

    print("👉 Bây giờ rê chuột đến góc dưới-phải và nhấn [F8]...")
    while True:
        if keyboard.is_pressed('f8'):
            p2 = get_mouse_pos()
            print(f" ✔ Điểm 2 (Góc dưới-phải): {p2}")
            time.sleep(0.5)
            break
        time.sleep(0.02)

    x1, y1 = p1
    x2, y2 = p2

    top_x = min(x1, x2)
    top_y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    if width < 5 or height < 5:
        print("[LỖI] Vùng chọn quá nhỏ (< 5px). Hãy thử lại.")
        return

    print("\n" + "-" * 50)
    print(f" 🎉 KẾT QUẢ VÙNG QUÉT (ROI):")
    print(f" Tọa độ X: {top_x}, Y: {top_y}, Rộng: {width}, Cao: {height}")
    print(f" Cấu hình điền vào config.json: [{top_x}, {top_y}, {width}, {height}]")
    print("-" * 50)

    # Chụp ảnh vùng này
    os.makedirs("assets", exist_ok=True)
    with mss.mss() as sct:
        monitor = {"top": top_y, "left": top_x, "width": width, "height": height}
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)[:, :, :3]

    print("\nBạn muốn lưu ảnh này thành file gì?")
    print(" 1. assets/B1_QuangCan/nut_quang_can.png (Bước 1: Nút Quăng Cần)")
    print(" 2. assets/B2_DangCau/nut_giat_can.png   (Bước 2: Nút Giật Cần)")
    print(" 3. assets/B2_DangCau/dau_cham_than/    (Bước 2: Dấu chấm than - lưu các màu)")
    print(" 4. assets/B2_DangCau/vat_neo_ten/      (Bước 2: Tên / Bảng tên nhân vật)")
    print(" 5. assets/B3_BaoQuan/nut_bao_quan.png   (Bước 3: Nút Bảo Quản cá)")
    print(" 6. Tự đặt tên khác")
    print(" 0. Bỏ qua không lưu")

    choice = input("Nhập lựa chọn (0-6): ").strip()
    target_path = None
    if choice == '1':
        os.makedirs("assets/B1_QuangCan", exist_ok=True)
        target_path = "assets/B1_QuangCan/nut_quang_can.png"
    elif choice == '2':
        os.makedirs("assets/B2_DangCau", exist_ok=True)
        target_path = "assets/B2_DangCau/nut_giat_can.png"
    elif choice == '3':
        folder = "assets/B2_DangCau/dau_cham_than"
        os.makedirs(folder, exist_ok=True)
        color_tag = input("Nhập tên màu hoặc ký hiệu (ví dụ: vang, do, xanh hoặc Enter): ").strip()
        if not color_tag:
            existing = len(os.listdir(folder)) + 1
            color_tag = f"mau_{existing}"
        target_path = os.path.join(folder, f"{color_tag}.png")
    elif choice == '4':
        folder = "assets/B2_DangCau/vat_neo_ten"
        os.makedirs(folder, exist_ok=True)
        name_tag = input("Nhập tên nhân vật (ví dụ: columbinaa, acc_phu...): ").strip()
        if not name_tag:
            name_tag = f"nhanvat_{len(os.listdir(folder)) + 1}"
        target_path = os.path.join(folder, f"{name_tag}.png")
    elif choice == '5':
        os.makedirs("assets/B3_BaoQuan", exist_ok=True)
        target_path = "assets/B3_BaoQuan/nut_bao_quan.png"
    elif choice == '6':
        name = input("Nhập tên file (ví dụ my_icon.png): ").strip()
        target_path = os.path.join("assets", name)

    if target_path:
        cv2.imwrite(target_path, img)
        print(f" ✅ ĐÃ LƯU THÀNH CÔNG VÀO: {target_path}\n")

def main():
    while True:
        print("\n" + "=" * 50)
        print("     TIỆN ÍCH ĐO TỌA ĐỘ & CHỤP ẢNH MẪU (HELPER)")
        print("=" * 50)
        print(" 1. Theo dõi tọa độ chuột trực tiếp (Real-time Inspector)")
        print(" 2. Chụp ảnh mẫu và lấy thông số vùng [F7 & F8]")
        print(" 0. Thoát ra")
        print("=" * 50)
        choice = input("Nhập lựa chọn (0-2): ").strip()

        if choice == '1':
            real_time_inspector()
        elif choice == '2':
            snip_and_save()
        elif choice == '0':
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại.")

if __name__ == "__main__":
    main()
