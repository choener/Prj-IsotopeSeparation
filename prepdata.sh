#!/usr/bin/env bash

# Prepare data for downstream analysis. Runs locally in parallel using the 'parallel' tool.

go() {
  echo $1
  ./IsotopeSep/prepdata.py -i $1
}

export -f go
parallel --jobs 20 go {} ::: ../_data/data/*

