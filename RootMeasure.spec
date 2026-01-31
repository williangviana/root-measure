# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Root Measure (GUI)

import os
import matplotlib
import customtkinter

block_cipher = None

a = Analysis(
    ['gui/app.py'],
    pathex=['scripts'],
    binaries=[],
    datas=[
        (matplotlib.get_data_path(), 'matplotlib/mpl-data'),
        (os.path.dirname(customtkinter.__file__), 'customtkinter'),
        ('scripts', 'scripts'),
    ],
    hiddenimports=[
        'customtkinter',
        'matplotlib.backends.backend_tkagg',
        'tkinter',
        'PIL',
        'skimage.morphology',
        'skimage.graph',
        'scipy.spatial',
        'scipy.stats',
        'networkx',
        'pandas',
        'tifffile',
        'cv2',
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio', 'functorch',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'IPython', 'jupyter', 'notebook', 'zmq',
        'sphinx', 'docutils', 'pytest',
        'scipy.io.matlab',
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RootMeasure',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='RootMeasure',
)

app = BUNDLE(
    coll,
    name='RootMeasure.app',
    icon='assets/RootMeasure.icns',
    bundle_identifier='com.dinnenylab.rootmeasure',
    info_plist={
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '1.0',
    },
)
