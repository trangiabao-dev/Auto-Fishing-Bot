"""
Path resolution utilities for AutoFishingBot.
Provides robust path finding for both development source mode and PyInstaller frozen executable mode.
"""
from __future__ import annotations

import os
import sys


def get_base_dir() -> str:
    """
    Returns the project root directory.
    When running as a PyInstaller bundle, returns the folder containing the executable.
    When running from source code, returns the project root (parent of 'core').
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_farm_assets_dir() -> str:
    """Returns the absolute path to assets/farm."""
    base = get_base_dir()
    # Check if assets/farm is directly under base
    target = os.path.join(base, "assets", "farm")
    if os.path.exists(target):
        return target
    # Fallback for PyInstaller internal layout if user packaged inside _internal
    internal_target = os.path.join(base, "_internal", "assets", "farm")
    if os.path.exists(internal_target):
        return internal_target
    return target


def get_farm_config_path() -> str:
    """Returns the absolute path to config/farm_config.json."""
    base = get_base_dir()
    return os.path.join(base, "config", "farm_config.json")


def get_digits_dir() -> str:
    """Returns the absolute path to digits templates."""
    return os.path.join(get_farm_assets_dir(), "digits")
