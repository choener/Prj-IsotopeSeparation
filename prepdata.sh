#!/usr/bin/env bash

TGT="../_data/carbon/prepared_data"

# Prepare data for downstream analysis. Runs locally in parallel using the 'parallel' tool.

go() {
  echo $TGT
  echo $1
  echo $2
  ./IsotopeSep/prepdata.py -o $3 --kmer $1 -i $2
}

export -f go
parallel go {} {} ::: 1 3 ::: ../_data/carbon/rs/* ::: $TGT

