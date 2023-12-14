#!/usr/bin/env bash

# Prepare data for downstream analysis. Runs locally in parallel using the 'parallel' tool.

go() {
  echo $1
  echo $2
  ./IsotopeSep/prepdata.py --kmer $1 -i $2
}

export -f go
parallel go {} {} ::: 1 3 5 ::: ../_data/data/*

