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
      barcodeZero="taubert-d2o/barcode14.ids"
      numZero="0.0"
      ;;
    30)
      barcodeZero="taubert-d2o/barcode15.ids"
      numZero="0.3"
      ;;
    100)
      barcodeZero="taubert-d2o/barcode16.ids"
      numZero="1.0"
      ;;
  esac
  case $5 in
    0)
      barcodeOne="taubert-d2o/barcode14.ids"
      numOne="0.0"
      ;;
    30)
      barcodeOne="taubert-d2o/barcode15.ids"
      numOne="0.3"
      ;;
    100)
      barcodeOne="taubert-d2o/barcode16.ids"
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
go ../_data/d2o/2024-08-25-mini-1-train 5 ../_data/d2o/cross-split-inputs-mini/1 0 30 TRAIN
go ../_data/d2o/2024-08-25-mini-2-test  5 ../_data/d2o/cross-split-inputs-mini/2 0 30 ../_data/d2o/2024-08-25-mini-1-train

# run data for 5-fold crossvalidation
# these are all dwell=no, partial=yes as per defaults

# 0-30 only

go ../_data/d2o/2024-08-25-A-cross-0-30/train-not-1 5 ../_data/d2o/cross-split-inputs-A/not-1 0 30 TRAIN
go ../_data/d2o/2024-08-25-A-cross-0-30/train-not-2 5 ../_data/d2o/cross-split-inputs-A/not-2 0 30 TRAIN
go ../_data/d2o/2024-08-25-A-cross-0-30/train-not-3 5 ../_data/d2o/cross-split-inputs-A/not-3 0 30 TRAIN
go ../_data/d2o/2024-08-25-A-cross-0-30/train-not-4 5 ../_data/d2o/cross-split-inputs-A/not-4 0 30 TRAIN
go ../_data/d2o/2024-08-25-A-cross-0-30/train-not-5 5 ../_data/d2o/cross-split-inputs-A/not-5 0 30 TRAIN

go ../_data/d2o/2024-08-25-A-cross-0-30/test-1 5 ../_data/d2o/cross-split-inputs-A/1 0 30 ../_data/d2o/2024-08-25-A-cross-0-30/train-not-1
go ../_data/d2o/2024-08-25-A-cross-0-30/test-2 5 ../_data/d2o/cross-split-inputs-A/2 0 30 ../_data/d2o/2024-08-25-A-cross-0-30/train-not-2
go ../_data/d2o/2024-08-25-A-cross-0-30/test-3 5 ../_data/d2o/cross-split-inputs-A/3 0 30 ../_data/d2o/2024-08-25-A-cross-0-30/train-not-3
go ../_data/d2o/2024-08-25-A-cross-0-30/test-4 5 ../_data/d2o/cross-split-inputs-A/4 0 30 ../_data/d2o/2024-08-25-A-cross-0-30/train-not-4
go ../_data/d2o/2024-08-25-A-cross-0-30/test-5 5 ../_data/d2o/cross-split-inputs-A/5 0 30 ../_data/d2o/2024-08-25-A-cross-0-30/train-not-5



#go ../_data/d2o/2024-08-25-B-cross-0-30/train-not-1 5 ../_data/d2o/cross-split-inputs-B/not-1 0 30 TRAIN
#go ../_data/d2o/2024-08-25-B-cross-0-30/train-not-2 5 ../_data/d2o/cross-split-inputs-B/not-2 0 30 TRAIN
#go ../_data/d2o/2024-08-25-B-cross-0-30/train-not-3 5 ../_data/d2o/cross-split-inputs-B/not-3 0 30 TRAIN
#go ../_data/d2o/2024-08-25-B-cross-0-30/train-not-4 5 ../_data/d2o/cross-split-inputs-B/not-4 0 30 TRAIN
#go ../_data/d2o/2024-08-25-B-cross-0-30/train-not-5 5 ../_data/d2o/cross-split-inputs-B/not-5 0 30 TRAIN

#go ../_data/d2o/2024-08-25-B-cross-0-30/test-1 5 ../_data/d2o/cross-split-inputs-B/1 0 30 ../_data/d2o/2024-08-25-B-cross-0-30/train-not-1
#go ../_data/d2o/2024-08-25-B-cross-0-30/test-2 5 ../_data/d2o/cross-split-inputs-B/2 0 30 ../_data/d2o/2024-08-25-B-cross-0-30/train-not-2
#go ../_data/d2o/2024-08-25-B-cross-0-30/test-3 5 ../_data/d2o/cross-split-inputs-B/3 0 30 ../_data/d2o/2024-08-25-B-cross-0-30/train-not-3
#go ../_data/d2o/2024-08-25-B-cross-0-30/test-4 5 ../_data/d2o/cross-split-inputs-B/4 0 30 ../_data/d2o/2024-08-25-B-cross-0-30/train-not-4
#go ../_data/d2o/2024-08-25-B-cross-0-30/test-5 5 ../_data/d2o/cross-split-inputs-B/5 0 30 ../_data/d2o/2024-08-25-B-cross-0-30/train-not-5



#go ../_data/d2o/2024-08-25-C-cross-0-30/train-not-1 5 ../_data/d2o/cross-split-inputs-C/not-1 0 30 TRAIN
#go ../_data/d2o/2024-08-25-C-cross-0-30/train-not-2 5 ../_data/d2o/cross-split-inputs-C/not-2 0 30 TRAIN
#go ../_data/d2o/2024-08-25-C-cross-0-30/train-not-3 5 ../_data/d2o/cross-split-inputs-C/not-3 0 30 TRAIN
#go ../_data/d2o/2024-08-25-C-cross-0-30/train-not-4 5 ../_data/d2o/cross-split-inputs-C/not-4 0 30 TRAIN
#go ../_data/d2o/2024-08-25-C-cross-0-30/train-not-5 5 ../_data/d2o/cross-split-inputs-C/not-5 0 30 TRAIN

#go ../_data/d2o/2024-08-25-C-cross-0-30/test-1 5 ../_data/d2o/cross-split-inputs-C/1 0 30 ../_data/d2o/2024-08-25-C-cross-0-30/train-not-1
#go ../_data/d2o/2024-08-25-C-cross-0-30/test-2 5 ../_data/d2o/cross-split-inputs-C/2 0 30 ../_data/d2o/2024-08-25-C-cross-0-30/train-not-2
#go ../_data/d2o/2024-08-25-C-cross-0-30/test-3 5 ../_data/d2o/cross-split-inputs-C/3 0 30 ../_data/d2o/2024-08-25-C-cross-0-30/train-not-3
#go ../_data/d2o/2024-08-25-C-cross-0-30/test-4 5 ../_data/d2o/cross-split-inputs-C/4 0 30 ../_data/d2o/2024-08-25-C-cross-0-30/train-not-4
#go ../_data/d2o/2024-08-25-C-cross-0-30/test-5 5 ../_data/d2o/cross-split-inputs-C/5 0 30 ../_data/d2o/2024-08-25-C-cross-0-30/train-not-5



# special runs

## given the 0-30 training data, how well do we test on others?
go ../_data/d2o/2024-08-25-A-cross-0-30-0-100/test-1 5 ../_data/d2o/cross-split-inputs-A/1 0 100 ../_data/d2o/2024-08-25-A-cross-0-30/train-not-1
go ../_data/d2o/2024-08-25-A-cross-0-30-100-30/test-1 5 ../_data/d2o/cross-split-inputs-A/1 100 30 ../_data/d2o/2024-08-25-A-cross-0-30/train-not-1

