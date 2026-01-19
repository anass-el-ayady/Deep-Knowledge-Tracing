# Knowledge Tracing Benchmark

This repository provides a unified framework to train, evaluate, and compare various Knowledge Tracing (KT) models on multiple datasets. It supports:

- Training a **single model** on all datasets
- Training **all models multiple times** (`n_runs`) to ensure robustness and account for randomness in deep learning

---

## Project Structure

- `dataloader.py`: Handles loading and preprocessing of KT datasets
- `evaluation.py`: Contains training, testing, and custom loss functions
- `n_runs.py`: Trains all models for multiple runs and aggregates results
- `run.py` (optional): Used to train a single model once
---

## Train a Single Model (Example: LSTM)

To train one model on all datasets:

```bash
python run.py --lstm --hidden=128 --epochs=10
```
## Train All Models with Multiple Runs
```bash
python n_runs.py --lstm --hidden=128 --epochs=10
```
Iterates through all model types (defined in model_types)
Runs each model n_runs times (default: 5)

## Why Use Multiple Runs?
Training deep learning models is stochastic: performance varies depending on random initialization, batch order, dropout, etc.

Running each model multiple times helps to:

  Reduce variance
  
  Report mean performance and standard deviation
  
  Make comparisons between models more reliable

## Dataset Format
Each .csv file should follow a three-line structure for each sequence:
```bash
<sequence_length>
<question_id_1>,<question_id_2>,...
<correctness_1>,<correctness_2>,...

```
