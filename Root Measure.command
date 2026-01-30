#!/bin/bash
cd "$(dirname "$0")/scripts"
PYTHONDONTWRITEBYTECODE=1 python3 measure_roots.py
