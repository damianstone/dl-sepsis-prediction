## Pre-processed dataset

- input dataset: fully imputed dataset
- total columns: 63
- columns dropped:

```bash
["Unit1", "Unit2", "cluster_id", "dataset", "HospAdmTime"]
```

- new columns added:

```bash
['SOFA',
 'NEWS',
 'qSOFA',
 'MAP_MA_3h',
 'MAP_SD_3h',
 'MAP_Delta',
 'MAP_MA_6h',
 'MAP_SD_6h',
 'MAP_MA_12h',
 'MAP_SD_12h',
 'Creatinine_MA_3h',
 'Creatinine_SD_3h',
 'Creatinine_Delta',
 'Creatinine_MA_6h',
 'Creatinine_SD_6h',
 'Creatinine_MA_12h',
 'Creatinine_SD_12h',
 'Platelets_MA_3h',
 'Platelets_SD_3h',
 'Platelets_Delta',
 'Platelets_MA_6h',
 'Platelets_SD_6h',
 'Platelets_MA_12h',
 'Platelets_SD_12h']
```

- forward + backward fill to handle nan values for the new columns added
