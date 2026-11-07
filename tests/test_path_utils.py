import os
import sys
import pytest
from core.path_utils import (
    get_base_dir,
    get_farm_assets_dir,
    get_farm_config_path,
    get_digits_dir,
)

def test_get_base_dir_normal(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    base = get_base_dir()
    assert os.path.exists(base)
    assert os.path.isdir(base)
    # Must contain 'core' and 'assets'
    assert os.path.exists(os.path.join(base, "core"))

def test_get_base_dir_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    fake_exe = tmp_path / "app.exe"
    fake_exe.touch()
    monkeypatch.setattr(sys, "executable", str(fake_exe))
    base = get_base_dir()
    assert base == str(tmp_path)

def test_farm_paths_exist_in_project():
    assets_dir = get_farm_assets_dir()
    assert os.path.exists(assets_dir), f"Farm assets dir should exist: {assets_dir}"
    assert os.path.exists(os.path.join(assets_dir, "thu_hoach"))
    assert os.path.exists(os.path.join(assets_dir, "ban"))

    digits_dir = get_digits_dir()
    assert os.path.exists(digits_dir), f"Digits dir should exist: {digits_dir}"

    cfg_path = get_farm_config_path()
    assert cfg_path.endswith("farm_config.json")
