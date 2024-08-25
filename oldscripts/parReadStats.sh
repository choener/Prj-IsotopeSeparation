#!/usr/bin/env bash

go() {
  echo $1
  ./IsotopeSep/ReadStats.py $1
}

export -f go
parallel --jobs 6 go {} {#} ::: taubert-d2o/reads/*fast5

