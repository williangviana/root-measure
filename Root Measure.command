#!/bin/bash
cd "$(dirname "$0")"
pip3 install -q -r requirements.txt
cd scripts
PYTHONDONTWRITEBYTECODE=1 python3 measure_roots.py
