## Pre-processed dataset

- input dataset: fully imputed dataset
- total columns: 97
- columns dropped:

```bash
["Unit1", "Unit2", "cluster_id", "dataset"]
```

- new columns added:

```bash
['SOFA_Creatinine',
 'SOFA_Platelets',
 'SOFA_Bilirubin_total',
 'SOFA_SaO2_FiO2',
 'SOFA_score',
 'NEWS_HR_score',
 'NEWS_Resp_score',
 'NEWS_Temp_score',
 'NEWS_SBP_score',
 'NEWS_O2Sat_score',
 'NEWS_FiO2_score',
 'NEWS_score',
 'qSOFA_Resp_score',
 'qSOFA_SBP_score',
 'qSOFA_score',
 'Shock_Index',
 'Bilirubin_Ratio',
 'HR_max_6h',
 'HR_min_6h',
 'HR_mean_6h',
 'HR_median_6h',
 'HR_std_6h',
 'HR_diff_std_6h',
 'O2Sat_max_6h',
 'O2Sat_min_6h',
 'O2Sat_mean_6h',
 'O2Sat_median_6h',
 'O2Sat_std_6h',
 'O2Sat_diff_std_6h',
 'SBP_max_6h',
 'SBP_min_6h',
 'SBP_mean_6h',
 'SBP_median_6h',
 'SBP_std_6h',
 'SBP_diff_std_6h',
 'MAP_max_6h',
 'MAP_min_6h',
 'MAP_mean_6h',
 'MAP_median_6h',
 'MAP_std_6h',
 'MAP_diff_std_6h',
 'Resp_max_6h',
 'Resp_min_6h',
 'Resp_mean_6h',
 'Resp_median_6h',
 'Resp_std_6h',
 'Resp_diff_std_6h',
 'HR_missing_count_global',
 'HR_missing_interval_mean_global',
 'O2Sat_missing_count_global',
 'O2Sat_missing_interval_mean_global',
 'SBP_missing_count_global',
 'SBP_missing_interval_mean_global',
 'MAP_missing_count_global',
 'MAP_missing_interval_mean_global',
 'Resp_missing_count_global',
 'Resp_missing_interval_mean_global']
```

- forward + backward fill to handle nan values for the new columns added
