## Transformers basic
* neural network for sequence data
* why not RNN ? struggle more with long sequences and irregular timing common in clinical data
* encoder = to classify or predict from a sequence 
* embedding = ssequence -> vectors
* self-attention = calculates the relative importance between elements in the sequence
* feed-forward = improves learned representation
* encoder block = [embedding -> self-attention -> feed-forward]
* transformer = stack of multiple encoder blocks
* input = only numeric vectors 

## Temporal Fusion Transformer (FTT)
* useful for irregular data (time series)
* separately handles static (age, gender) and temporal variables for improved predictions

## Custom dataset
* padding -> adding extra values (zeros) to make all the sequences the same length, allow consistent shape
* we can add `masking` to tell the model to ignore padded values and prevent noise 
```python
Patient A records: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  (Length 2)  
Patient B records: [[0.7, 0.8, 0.9]]  (Length 1)  

After
Patient A: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  
Patient B: [[0.7, 0.8, 0.9], [0.0, 0.0, 0.0]]  (extra row added)  
```
* 

## Metrics
* AUC-ROC: tells how well your model separates patients with and without sepsis
* precision, Recall, F1: tell how many sepsis cases your model finds correctly and incorrectly
* ROC curve: shows clearly if your model makes few or many false alarms.
* precision-recall curve: Shows if your model misses many real sepsis cases
* attention heatmaps: shows which patient features matter most

## After training 
* which variables are more important
* how long before it detects sepsis
* how influencial are the least of sequences

## Small dataset creation ideas
* only 1.80% positive sepsis -> possible data imbalance for models
* simple 50/50 sampling (for now)
* stratified sampling, get x amount of positives and x amount of negatives form the big dataset

### 01_simple_transformer evaluation
* trained with the original dataset (imputed_sofa) 80/20
* epochs = 100
* precision 98% because most of the data were true negatives 
#### Covariance matrix
* high TN -> correctly classifies non-sepsis 
* high FN -> not predicting correctly for sepsis patients 

#### Precision - recall
* precision drops as recall increases, meaning that capturing more positives leads to more false positives due to dataset imbalance

### 03_simple_transformer evaluation
* trained with X_train, y_train as balance dataset 80/20
* tested with X_test, y_test as unbalance (original) dataset
* 100 epochs
* 77.69% acc on test data
