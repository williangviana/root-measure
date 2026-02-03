"""cx_Freeze setup for Root Measure .app bundle (macOS) or .exe (Windows)."""

import os
import sys
from pathlib import Path
from cx_Freeze import setup, Executable

# Use Path objects for cross-platform compatibility
scripts_dir = Path("scripts")
gui_dir = Path("gui")

build_options = {
    "packages": [
        "customtkinter",
        "cv2",
        "numpy",
        "scipy",
        "skimage",
        "pandas",
        "tifffile",
        "matplotlib",
        "PIL",
        "tkinter",
    ],
    "excludes": [
        "torch", "torchvision", "torchaudio",
        "PyQt5", "PyQt6", "PySide2", "PySide6",
        "IPython", "jupyter", "notebook",
        "pytest", "sphinx",
    ],
    "include_files": [
        (str(scripts_dir), str(Path("lib") / "scripts")),
        (str(gui_dir / "canvas.py"), str(Path("lib") / "gui" / "canvas.py")),
        (str(gui_dir / "sidebar.py"), str(Path("lib") / "gui" / "sidebar.py")),
        (str(gui_dir / "workflow.py"), str(Path("lib") / "gui" / "workflow.py")),
    ],
    "path": sys.path + [str(scripts_dir), str(gui_dir)],
}

bdist_mac_options = {
    "bundle_name": "Root Measure",
    "codesign_identity": "-",
    "codesign_deep": True,
}

# Platform-specific executable settings
if sys.platform == "win32":
    # Windows: use Win32GUI base to hide console window
    exe_base = "Win32GUI"
    target_name = "RootMeasure.exe"
else:
    # macOS/Linux
    exe_base = "gui"
    target_name = "RootMeasure"

setup(
    name="RootMeasure",
    version="1.0.0",
    description="Measure root lengths from scanned agar plate images",
    author="Willian Viana",
    options={
        "build_exe": build_options,
        "bdist_mac": bdist_mac_options,
    },
    executables=[
        Executable(
            str(gui_dir / "app.py"),
            base=exe_base,
            target_name=target_name,
        )
    ],
)
