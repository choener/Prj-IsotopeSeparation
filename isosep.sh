#!/usr/bin/env bash

MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 ./isosep.py $@

