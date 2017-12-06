#!/bin/bash
set -e
python scripts/download.py

sh build_java.sh

python scripts/preprocess_hs.py
python scripts/preprocess_django.py