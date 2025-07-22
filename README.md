- Download dataset: https://archive.physionet.org/users/shared/challenge-2019/
- More info about the challenge: https://physionet.org/content/challenge-2019/1.0.0/

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
<img width="700" alt="dataset_versions drawio (1)" src="https://github.com/user-attachments/assets/49b2b135-7987-4264-933b-7ee1c047d68a" />

### Trasformer - hour by hour prediction
<img width="500" alt="time-step-transformer drawio" src="https://github.com/user-attachments/assets/758f3a2e-f37c-4693-8ef5-a43a29893f8c" />


### Transformer - patient level prediction
<img width="500" alt="sepsis-transformer drawio (1)" src="https://github.com/user-attachments/assets/6738bb78-cbec-4b1e-aaed-107c81763d0e" />
