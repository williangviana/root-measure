"""cx_Freeze setup for Root Measure macOS .app bundle."""

import sys
from cx_Freeze import setup, Executable

build_options = {
    "packages": [
        "customtkinter",
        "cv2",
        "numpy",
        "scipy",
        "skimage",
        "networkx",
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
        ("scripts", "lib/scripts"),
        ("gui/canvas.py", "lib/gui/canvas.py"),
        ("gui/sidebar.py", "lib/gui/sidebar.py"),
        ("gui/workflow.py", "lib/gui/workflow.py"),
    ],
    "path": sys.path + ["scripts", "gui"],
}

bdist_mac_options = {
    "bundle_name": "Root Measure",
    "codesign_identity": "-",
    "codesign_deep": True,
}

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
            "gui/app.py",
            base="gui",
            target_name="RootMeasure",
        )
    ],
)
