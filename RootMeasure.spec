# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Root Measure

import os
import matplotlib

block_cipher = None

a = Analysis(
    ['scripts/measure_roots.py'],
    pathex=['scripts'],
    binaries=[],
    datas=[
        (matplotlib.get_data_path(), 'matplotlib/mpl-data'),
    ],
    hiddenimports=[
        'matplotlib.backends.backend_tkagg',
        'tkinter',
        'PIL',
        'skimage.morphology',
        'skimage.graph',
        'scipy.spatial',
        'networkx',
        'pandas',
        'tifffile',
        'cv2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    console=True,
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
    bundle_identifier='com.dinnenylab.rootmeasure',
)
