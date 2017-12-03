#!/bin/bash
set -e
python3 scripts/download.py

sh build_java.sh

python3 scripts/preprocess_hs.py