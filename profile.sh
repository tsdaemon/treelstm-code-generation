#!/usr/bin/env bash
time python -m cProfile -o valid.prof main.py -dataset hs -no_cuda -mode validate -data_dir ./preprocessed/hs/unary_closures -output_dir ./results/hs