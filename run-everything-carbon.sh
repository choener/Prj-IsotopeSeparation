#!/usr/bin/env bash

# 2024-08-25
# choener
# This script runs all data. It uses small check-pointing files "done-???". If
# they exist in a directory, nothing is done. If they are removed the work for
# that directory is repeated. There might be multiple done-files, depending on
# the types of work for each directory. In some cases, the done files are
# different.

# Train the model for this directory.
# Checks for existence of an adagrad-trace.netcdf file.

# train models [unless they are trained already]
# $1 is the target directory
# $2 the kmer length [1,3,5]
# $3 the prepared_data dir to work on
# $4, $5 the zero and one targets [0.0 0.3 1.0]
# $6 is either TRAIN or the directory to take the netcdf file from

function go {
  mkdir -p $1
  # which barcode?
  barcodeZero=""
  numZero=""
  barcodeOne=""
  numOne=""

  case $4 in
    0)
      barcodeZero="../_data/carbon/barcodes/barcode20.ids"
      numZero="0.0"
      ;;
    100)
      barcodeZero="../_data/carbon/barcodes/barcode21.ids"
      numZero="1.0"
      ;;
  esac
  case $5 in
    0)
      barcodeOne="../_data/carbon/barcodes/barcode20.ids"
      numOne="0.0"
      ;;
    100)
      barcodeOne="../_data/carbon/barcodes/barcode21.ids"
      numOne="1.0"
      ;;
  esac

  # run (maybe)
  # we train if $6 is TRAIN, and the netcdf file does not exist
  if [[ ! -f ./${1}/${2}-adagrad-trace.netcdf && "${6}" == "TRAIN" ]]
  then
    echo "training model"
    ./IsotopeSep/isosep.py \
      --outputdir $1 \
      --kmer $2 \
      --inputdirs $3 \
      --zero $numZero \
      --one $numOne \
      --barcode $4 $barcodeZero \
      --barcode $5 $barcodeOne \
      --sampler adagrad \
      --posteriorpredictive \
      --train
  else
    echo "no training necessary"
  fi

  # copy netcdf file if necessary
  if [[ ${6} != "TRAIN" ]]
  then
    cp -a ${6}/${2}-adagrad-trace.netcdf ${1}
  fi

  # in case the csv's are not there, run everything, except training
  if [[ -f ./${1}/${2}-adagrad-trace.netcdf && ! -f ./${1}/${2}-adagrad-fdr.csv ]]
  then
    echo "running posterior predictions"
    ./IsotopeSep/isosep.py \
      --outputdir $1 \
      --kmer $2 \
      --inputdirs $3 \
      --zero $numZero \
      --one $numOne \
      --barcode $4 $barcodeZero \
      --barcode $5 $barcodeOne \
      --sampler adagrad \
      --posteriorpredictive
  else
    echo "no posterior predictions necessary"
  fi
}


export -f go

#parallel --jobs 1 train {} {} {} {} {} ::: 1 2 3 4 5

# (fast) tests on small inputs; not super fast but faster than on all data
go ../_data/carbon/2024-08-25-mini-train 5 ../_data/carbon/mini-data     0 100 TRAIN

go ../_data/carbon/2024-08-25-train      5 ../_data/carbon/prepared_data 0 100 TRAIN

# NOTE: if carbon worked, we'd test now
#
# go ../_data/carbon/2024-08-25-test 5 ../_data/carbon/prepared-data 0 100 ../_data/carbon/2024-08-25-train

