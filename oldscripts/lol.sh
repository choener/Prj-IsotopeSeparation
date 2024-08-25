#!/usr/bin/env bash

go() {
  echo $1
  mkdir carbon-out/`basename $1 .fast5`
  ./IsotopeSep/ReadStats.py --outdir carbon-out/`basename $1  .fast5` $1 2>/dev/null
}

export -f go
parallel --jobs 6 go {} {#} ::: ./taubert-carbon/reads/*.fast5

