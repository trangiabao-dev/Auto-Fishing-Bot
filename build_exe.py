import os
import sys
import shutil
import subprocess

def build():
    print("🚀 Bắt đầu đóng gói AutoFishingBot sang .exe (Bản tối ưu dung lượng)...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # PyInstaller chỉ đóng gói runtime code, không nhúng trùng lặp assets vào _internal
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", "AutoFishingBot",
        "--hidden-import", "pynput.keyboard._win32",
        "--hidden-import", "pynput.mouse._win32",
        "--hidden-import", "mss",
        "--hidden-import", "cv2",
        "app_gui.py"
    ]
    
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("❌ Lỗi khi đóng gói PyInstaller!")
        return res.returncode

    dist_dir = os.path.join(base_dir, "dist", "AutoFishingBot")
    dst_assets = os.path.join(dist_dir, "assets")
    dst_config = os.path.join(dist_dir, "config.json")

    # Loại bỏ thư viện video ffmpeg nặng ~29.5MB không dùng tới (bot chỉ xử lý ảnh tĩnh)
    ffmpeg_dll = os.path.join(dist_dir, "_internal", "cv2", "opencv_videoio_ffmpeg500_64.dll")
    if os.path.exists(ffmpeg_dll):
        try:
            os.remove(ffmpeg_dll)
            print("🧹 Đã loại bỏ opencv_videoio_ffmpeg500_64.dll (tiết kiệm ~29.5MB).")
        except Exception as e:
            print(f"⚠️ Không thể xóa ffmpeg dll: {e}")

    # Đảm bảo assets và config.json nằm trực tiếp cùng cấp với AutoFishingBot.exe để người dùng dễ tinh chỉnh
    if os.path.exists(dst_assets):
        shutil.rmtree(dst_assets)

    # Chỉ copy các template thực tế cần thiết, lọc bỏ ảnh mẫu test, archive và ảnh debug
    def ignore_junk(folder, contents):
        ignored = []
        for c in contents:
            if c in ("raw_samples", "thong bao tui da day", "tin nhan tui da day", "archive", "__pycache__"):
                ignored.append(c)
            elif "farm" not in folder and (c.endswith(".jpg") or c == "anh_man_hinh_bot_thay.png"):
                ignored.append(c)
        return ignored

    shutil.copytree(os.path.join(base_dir, "assets"), dst_assets, ignore=ignore_junk)
    shutil.copy2(os.path.join(base_dir, "config.json"), dst_config)

    # Đảm bảo thư mục config có sẵn cho farm_config.json
    dst_cfg_dir = os.path.join(dist_dir, "config")
    os.makedirs(dst_cfg_dir, exist_ok=True)
    src_farm_cfg = os.path.join(base_dir, "config", "farm_config.json")
    if os.path.exists(src_farm_cfg):
        shutil.copy2(src_farm_cfg, os.path.join(dst_cfg_dir, "farm_config.json"))

    # Đảm bảo thư mục debug_hut_ca có sẵn cấu trúc 2 thư mục con (anh & debug)
    os.makedirs(os.path.join(dist_dir, "debug_hut_ca", "anh"), exist_ok=True)
    os.makedirs(os.path.join(dist_dir, "debug_hut_ca", "debug"), exist_ok=True)

    # Nén file zip phân phối
    zip_out = os.path.join(base_dir, "dist", "AutoFishingBot-Standalone")
    zip_file = zip_out + ".zip"
    if os.path.exists(zip_file):
        os.remove(zip_file)
    shutil.make_archive(zip_out, 'zip', os.path.dirname(dist_dir), os.path.basename(dist_dir))

    print(f"\n✅ ĐÓNG GÓI THÀNH CÔNG (TỐI ƯU DUNG LƯỢNG)!")
    print(f"📁 Thư mục chạy trực tiếp: {dist_dir}")
    print(f"📦 File nén mang đi máy khác: {zip_file} ({os.path.getsize(zip_file) / (1024*1024):.1f} MB)")
    return 0

if __name__ == "__main__":
    raise SystemExit(build())
