# Prj-IsotopeSeparation
Stats model that separates isotopically labelled ONT data

Meta-repository that collects the diverse parts needed for isotope separation.

Currently:
- stats model: <https://github.com/choener/IsotopeSeparation>
- "small data"
- "scripts to create links to big data" (big is everything that I can't keep on this tiny, local ssd)

# Running things

1. prepare data
1. run model:
    ```
    ./IsotopeSep/isosep.py \
      --barcode 0 taubert-d2o/barcode14.ids \
      --barcode 100 taubert-d2o/barcode16.ids \
      --outputdir fulltmp \
      --pickledreads pickles/50 \
      --kmer k5 \
      --posteriorpredictive \
      --train
    ```
