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

# Performance tips

- Keep the ``store`` directory on the local machine, where it can be accessed quickly. The summary
  statistics inputs files won't be needed that often in comparison and can be stored on a network
  drive.

# TODO

- Currently the ``barcode*.ids`` is generated from ``cat eventalign_summary.csv | awk '{print $2}' > barcode.ids``. This should be improved.

# Barcode associations

| Percent | Isotope | Code          |
|---------|---------|---------------|
| 0       | D2O     | barcode14.ids |
| 30      | D20     | barcode15.ids |
| 100     | D20     | barcode16.ids |

