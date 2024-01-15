#!/usr/bin/env bash

# run the isosep program with certain parameters
# arg 1: cross validation arguments
# arg 2: either 30 or 100%
# arg 3: no dwell or dwell
# arg 4: only full reads, allow partial reads
go() {
  notO="[2,3,4,5]"
  notT="[1,3,4,5]"
  notH="[1,2,4,5]"
  notF="[1,2,3,5]"
  notI="[1,2,3,4]"
  notx=""
  case $1 in
    1)
      notx=$notO
      ;;
    2)
      notx=$notT
      ;;
    3)
      notx=$notH
      ;;
    4)
      notx=$notF
      ;;
    5)
      notx=$notI
      ;;
  esac
  barcodeOne=taubert-d2o/barcode15.ids
  barcodeNum="0.3"
  if [[ $2 == 100 ]]
  then
    barcodeOne=taubert-d2o/barcode16.ids
    barcodeNum="1.0"
  fi
  dwellpart="dwellNO"
  if [[ $3 == "--dwell" ]]
  then
    dwellpart="dwellYES"
  fi
  partialpart="partialYES"
  if [[ $4 == "--onlycomplete" ]]
  then
    partialpart="partialNO"
  fi
  trainname="cross-0-${2}-${dwellpart}-${partialpart}/train-not${1}"
  testname="cross-0-${2}-${dwellpart}-${partialpart}/test-is${1}"
  mkdir -p $trainname
  mkdir -p $testname
  echo $1
  echo $2
  echo $3
  echo $4
  # train only if netcdf does not exist
  if [[ ! -f "${trainname}/5-adagrad-trace.netcdf" ]]
  then
    ./IsotopeSep/isosep.py \
      --barcode 0 taubert-d2o/barcode14.ids \
      --barcode $2 $barcodeOne \
      --inputdirs ../_data/d2o/crossvalidation/$notx \
      --zero 0.0 \
      --one $barcodeNum \
      --kmer 5 \
      --sampler adagrad \
      --posteriorpredictive \
      --train \
      --outputdir "${trainname}" \
      $3 $4
  else
    echo "already done"
  fi
  # test only if netcdf does not exist
  if [[ ! -f "${testname}/5-adagrad-trace.netcdf" ]]
  then
    cp "${trainname}/5-adagrad-trace.netcdf" "${testname}/5-adagrad-trace.netcdf"
    ./IsotopeSep/isosep.py \
      --barcode 0 taubert-d2o/barcode14.ids \
      --barcode $2 $barcodeOne \
      --inputdirs ../_data/d2o/crossvalidation/[1] \
      --zero 0.0 \
      --one $barcodeNum \
      --kmer 5 \
      --sampler adagrad \
      --posteriorpredictive \
      --outputdir "${testname}" \
      $3 $4
  else
    echo "already done"
  fi
}

export -f go

parallel --jobs 1 go {} {} {} {} ::: 1 2 3 4 5 ::: 30 100 ::: "" "--dwell" ::: "" "--onlycomplete"

#./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[2,3,4,5] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-2345
#./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[1,3,4,5] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-1345
#./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[1,2,4,5] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-1245
#./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[1,2,3,5] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-1235
#./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[1,2,3,4] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-1234

#./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[2] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --outputdir cross-0-100/test-2
#./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[3] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --outputdir cross-0-100/test-3
#./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[4] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --outputdir cross-0-100/test-4
#./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[5] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --outputdir cross-0-100/test-5
