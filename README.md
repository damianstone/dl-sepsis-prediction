- download dataset: https://archive.physionet.org/users/shared/challenge-2019/
- more info about the challenge: https://physionet.org/content/challenge-2019/1.0.0/

#### Create a virtual environment

```python
python3 -m venv venv-sepsis
source venv-sepsis/bin/activate
```

#### Install requirements.txt

```python
pip install -r requirements.txt
```

#### Setup Pre-commit hooks for linting

```
pre-commit clean
pre-commit install
```

### Datasets (after feature engineering)

1. Full imbalanced
2. Downsampled 70/30
3. Oversampled 70/30
