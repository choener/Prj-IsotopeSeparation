#!/usr/bin/env bash

./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[2,3,4,5] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-2345
./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[1,3,4,5] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-1345
./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[1,2,4,5] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-1245
./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[1,2,3,5] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-1235
./IsotopeSep/isosep.py --barcode 0 taubert-d2o/barcode14.ids --barcode 100 taubert-d2o/barcode16.ids --inputdirs ../_data/d2o/crossvalidation/[1,2,3,4] --zero 0.0 --one 1.0 --kmer 5 --sampler adagrad --posteriorpredictive --train --outputdir cross-0-100/train-1234
