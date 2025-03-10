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

## Metrics
* AUC-ROC: tells how well your model separates patients with and without sepsis
* precision, Recall, F1: tell how many sepsis cases your model finds correctly and incorrectly
* ROC curve: shows clearly if your model makes few or many false alarms.
* precision-recall curve: Shows if your model misses many real sepsis cases
* confusion matrix: shows exactly which mistakes your model made
* attention heatmaps: shows which patient features matter most

## After training 
* which variables are more important
* how long before it detects sepsis
* how influencial are the least of sequences 