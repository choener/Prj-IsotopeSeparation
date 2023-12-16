#!/usr/bin/env bash

go() {
  echo $1
  echo $2
  D=`basename $1 .fast5`
  mkdir -p rs/$D
  ./IsotopeSep/ReadStats.py --outdir rs/$D $1
}

export -f go
parallel go {} {#} ::: ../_data/d2o/raw_data/*.fast5

