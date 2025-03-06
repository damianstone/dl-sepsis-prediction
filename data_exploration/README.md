### Summary of hospital data
Hospital system: A
  - Number of patients: 20336
  - Number of septic patients: 1790
  - Sepsis prevalence: 8.8%
  - Number of rows: 790215
  - Number of entries: 11876446
  - Density of entries: 35.0%

Hospital system: B
  - Number of patients: 20000
  - Number of septic patients: 1142
  - Sepsis prevalence: 5.7%
  - Number of rows: 761995
  - Number of entries: 11356429
  - Density of entries: 34.7%

### Columns
* all -> ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
'HospAdmTime', 'ICULOS', 'SepsisLabel', 'patient_id', 'dataset']
* not continuous values -> ['Gender', 'Unit1', 'Unit2', 'dataset']
* target variable -> 'SepsisLabel'

### Other stuff
* patients have different numbers of records
