## Transformers basic

- neural network for sequence data
- why not RNN ? struggle more with long sequences and irregular timing common in clinical data
- encoder = to classify or predict from a sequence
- embedding = ssequence -> vectors
- self-attention = calculates the relative importance between elements in the sequence
- feed-forward = improves learned representation
- encoder block = [embedding -> self-attention -> feed-forward]
- transformer = stack of multiple encoder blocks
- input = only numeric vectors

## Encoder Transformer

- sequence: `[batch, time, features]`
- compares each moment in a patient’s timeline to all others, finding important changes, then predicts sepsis based on those patterns
- only uses the last moment to decide sepsis, which may ignore early important signs, this is why we use time series transformer

## Time Series Transformer (http://youtube.com/watch?v=30d8dFHuxf0)

- `input embedding` -> transformer features into a higher dimension for better learning
- `position encoding` -> add number to each time step (patient records) helping the model to know the order of events
- `encoder layers` -> learn relationships between all time steps using attention. n_heads > 1 is multi attention
- `global pooling` -> sumarises all time steps into one fixed-size vector
- `classification head` -> output final prediction, in this a linea layer for binary classification
- self-attention: each patient’s record looks at all past records using one way of thinking. It finds just one type of pattern
- multi-attention: each record looks at others in several ways at once. some focus on recent changes, others on older events
