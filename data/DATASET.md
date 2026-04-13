# BCCC-CSE-CIC-IDS2018 Dataset

## Dataset Source

This project uses the **BCCC-CSE-CIC-IDS2018** dataset, an updated version of the CSE-CIC-IDS2018 network intrusion detection dataset.
CSE-CIC-IDS2018 is approx 500GB of raw network packet capture data as PCAP files, hosted free on AWS
https://registry.opendata.aws/cse-cic-ids2018/

CICFlowMeter is a tool which groups packets into flows, and processes it into features for machine learning. The dataset created is about 7GB, which fits locally in RAM for testing.

The updated dataset is BCCC-CSE-CIC-IDS2018, which fixes some bugs in the original processed dataset and provides more features. This uses the original PCAP raw data, but the new processed dataset is about 100GB.


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

### To create sample of CSE-CIC-IDS2018
```
head -1 data/raw/NF-CICIDS2018-v3.csv > data/raw/NF-CICIDS2018-v3-sample.csv
tail -n +2 data/raw/NF-CICIDS2018-v3.csv | head -c 524288000 | sed '$d' >> data/raw/NF-CICIDS2018-v3-sample.csv
```