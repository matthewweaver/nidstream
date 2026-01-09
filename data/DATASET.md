# BCCC-CSE-CIC-IDS2018 Dataset

## Dataset Source

This project uses the **BCCC-CSE-CIC-IDS2018** dataset, an updated version of the CSE-CIC-IDS2018 network intrusion detection dataset.

**Download Link:** https://www.kaggle.com/datasets/bcccdatasets/large-scale-ids-dataset-bccc-cse-cic-ids2018?resource=download

### To create sample for local exploration
```
(head -n 1 friday_02_03_2018_benign.csv && \
 grep -E "2018-03-02 09:(1[5-9]|[2-3][0-9]|4[0-5]):" friday_02_03_2018_benign.csv) \
 > friday_02_03_2018_benign_sample.csv
```

```
(head -n 1 friday_02_03_2018_bot.csv && \
 grep -E "2018-03-02 09:(1[5-9]|[2-3][0-9]|4[0-5]):" friday_02_03_2018_bot.csv) \
 > friday_02_03_2018_bot_sample.csv
```

```
(head -n 1 friday_02_03_2018_benign_sample.csv && \
 tail -n +2 friday_02_03_2018_benign_sample.csv && \
 tail -n +2 friday_02_03_2018_bot_sample.csv) \
 > friday_02_03_2018_combined_sample.csv
```

### To create combined for training/testing
```
head -n 1 friday_02_03_2018_benign.csv && tail -n +2 friday_02_03_2018_benign.csv && tail -n +2 friday_02_03_2018_bot.csv > friday_02_03_2018_combined.csv
```