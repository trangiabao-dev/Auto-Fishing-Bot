import os
import sys
import shutil
import subprocess

def build():
    print("🚀 Bắt đầu đóng gói AutoFishingBot sang .exe...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", "AutoFishingBot",
        "--add-data", "assets;assets",
        "--add-data", "config.json;.",
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

    # Đảm bảo assets và config.json nằm trực tiếp cùng cấp với AutoFishingBot.exe để người dùng dễ tinh chỉnh
    if not os.path.exists(dst_assets):
        shutil.copytree(os.path.join(base_dir, "assets"), dst_assets)
    if not os.path.exists(dst_config):
        shutil.copy2(os.path.join(base_dir, "config.json"), dst_config)

    # Nén file zip phân phối
    zip_out = os.path.join(base_dir, "dist", "AutoFishingBot-Standalone")
    shutil.make_archive(zip_out, 'zip', os.path.dirname(dist_dir), os.path.basename(dist_dir))

    zip_file = zip_out + ".zip"
    print(f"\n✅ ĐÓNG GÓI THÀNH CÔNG!")
    print(f"📁 Thư mục chạy trực tiếp: {dist_dir}")
    print(f"📦 File nén mang đi máy khác: {zip_file} ({os.path.getsize(zip_file) / (1024*1024):.1f} MB)")
    return 0

if __name__ == "__main__":
    raise SystemExit(build())
