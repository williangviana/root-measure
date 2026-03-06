"""cx_Freeze setup for Root Measure .app bundle (macOS) or .exe (Windows)."""

import os
import sys
from pathlib import Path
from cx_Freeze import setup, Executable

# Use Path objects for cross-platform compatibility
scripts_dir = Path("scripts")
gui_dir = Path("gui")
icon_dir = Path("icon")

# On Windows, numpy 2.x ships DLLs in a separate numpy.libs/ directory.
# cx_Freeze needs bin_path_includes to find them during dependency analysis.
_bin_path_includes = []
if sys.platform == "win32":
    try:
        import numpy as _np
        np_dir = Path(_np.__file__).parent
        for dll_dir in [np_dir / ".libs", np_dir.parent / "numpy.libs"]:
            if dll_dir.is_dir():
                _bin_path_includes.append(str(dll_dir))
        # Also include scipy.libs if present
        import scipy as _sp
        sp_dir = Path(_sp.__file__).parent
        for dll_dir in [sp_dir / ".libs", sp_dir.parent / "scipy.libs"]:
            if dll_dir.is_dir():
                _bin_path_includes.append(str(dll_dir))
    except Exception:
        pass

build_options = {
    "packages": [
        "customtkinter",
        "cv2",
        "numpy",
        "numpy._core",
        "numpy.core",
        "scipy",
        "skimage",
        "pandas",
        "tifffile",
        "matplotlib",
        "PIL",
        "tkinter",
        "statsmodels",
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
    "bin_path_includes": _bin_path_includes,
    "path": sys.path + [str(scripts_dir), str(gui_dir)],
}

bdist_mac_options = {
    "bundle_name": "Root Measure",
    "iconfile": str(icon_dir / "icon.icns"),
    "codesign_identity": "-",
    "codesign_deep": True,
}

# Platform-specific executable settings
if sys.platform == "win32":
    # Windows: use Win32GUI base to hide console window
    exe_base = "Win32GUI"
    target_name = "RootMeasure.exe"
    icon_file = str(icon_dir / "icon.ico")
else:
    # macOS/Linux
    exe_base = "gui"
    target_name = "RootMeasure"
    icon_file = str(icon_dir / "icon.icns")

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
            icon=icon_file,
        )
    ],
)
