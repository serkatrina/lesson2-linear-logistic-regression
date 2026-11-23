"""Тесты для homework_experiments.py и для связки датасетов с обучением."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import homework_datasets as hd
import homework_experiments as he
import homework_model_modification as hm


def make_csv(tmp_path, n=150, seed=0):
    """Маленький CSV с регрессионной и классификационной целевой переменной."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    flag = np.where(rng.random(n) < 0.5, "no", "yes")
    target = 2 * a - b + 0.5 * (flag == "yes") + rng.normal(0, 0.1, n)
    label = np.where(a + b > 0, "pos", "neg")
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"a": a, "b": b, "flag": flag, "target": target, "label": label}).to_csv(path, index=False)
    return path


def regression_splits(tmp_path):
    return hd.load_csv_splits(make_csv(tmp_path), "target", "regression", drop_columns=["label"])


def classification_splits(tmp_path):
    return hd.load_csv_splits(make_csv(tmp_path), "label", "binary", drop_columns=["target"])


# --------------------------- признаки (feature engineering) ---------------------------
def test_polynomial_features():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    block, names = he.polynomial_features(X, ["a", "b"], [0, 1])
    assert torch.equal(block, torch.tensor([[1.0, 4.0], [9.0, 16.0]]))
    assert names == ["a^2", "b^2"]
    block, names = he.polynomial_features(X, ["a", "b"], [0, 1], degree=3)
    assert tuple(block.shape) == (2, 4) and names == ["a^2", "b^2", "a^3", "b^3"]
    block, names = he.polynomial_features(X, ["a", "b"], [])        # нет числовых признаков
    assert tuple(block.shape) == (2, 0) and names == []


def test_interaction_features():
    X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    block, names = he.interaction_features(X, ["a", "b", "c"])
    assert names == ["a*b", "a*c", "b*c"]
    assert torch.equal(block, torch.tensor([[2.0, 3.0, 6.0], [6.0, 8.0, 12.0]]))
    block, names = he.interaction_features(X[:, :1], ["a"])          # один признак: пар нет
    assert tuple(block.shape) == (2, 0) and names == []


def test_statistical_features():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    block, names = he.statistical_features(X, [0, 1])
    assert names == ["row_mean", "row_var"]
    assert torch.allclose(block, torch.tensor([[1.5, 0.25], [3.5, 0.25]]))


def test_engineer_datasets_shapes_and_standardization(tmp_path):
    train, val, test = regression_splits(tmp_path)               # признаки: a, b, flag
    assert train.input_dim == 3
    train2, val2, test2 = he.engineer_datasets(train, val, test, ("poly", "inter", "stat"))
    # 3 исходных + 2 квадрата (a, b) + 3 взаимодействия + 2 статистики
    assert train2.input_dim == val2.input_dim == test2.input_dim == 10
    assert len(train2.feature_names) == 10
    new_block = train2.X[:, 3:]
    assert torch.allclose(new_block.mean(dim=0), torch.zeros(7), atol=1e-4)
    spread = ((new_block - new_block.mean(dim=0)) ** 2).mean(dim=0).sqrt()
    assert torch.allclose(spread, torch.ones(7), atol=1e-3)
    assert train.input_dim == 3                                  # исходные датасеты не изменились


def test_engineer_datasets_without_methods_returns_same_objects(tmp_path):
    train, val, test = regression_splits(tmp_path)
    assert he.engineer_datasets(train, val, test, ()) == (train, val, test)


# --------------------------------- обучение на датасетах ---------------------------------
def test_dataloader_batches_from_csv_dataset(tmp_path):
    train, _, _ = regression_splits(tmp_path)
    X_batch, y_batch = next(iter(DataLoader(train, batch_size=8)))
    assert tuple(X_batch.shape) == (8, 3) and tuple(y_batch.shape) == (8, 1)
    train_c, _, _ = classification_splits(tmp_path)
    X_batch, y_batch = next(iter(DataLoader(train_c, batch_size=8)))
    assert tuple(y_batch.shape) == (8, 1)


def test_run_training_regression(tmp_path):
    train, val, test = regression_splits(tmp_path)
    result = hm.run_training(train, val, test, optimizer_name="Adam", lr=0.05, batch_size=32,
                             epochs=40, patience=5, log_every=1000)
    assert set(result["metrics"]) == {"mse", "rmse", "r2"}
    assert result["metrics"]["r2"] > 0.9                         # зависимость почти линейная
    assert len(result["history"]["train_loss"]) <= 40
    assert result["seconds"] >= 0


def test_run_training_classification(tmp_path):
    train, val, test = classification_splits(tmp_path)
    result = hm.run_training(train, val, test, optimizer_name="Adam", lr=0.05, batch_size=32,
                             epochs=60, log_every=1000)
    assert set(result["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert result["metrics"]["accuracy"] > 0.85 and result["metrics"]["roc_auc"] > 0.9


def test_run_training_is_reproducible(tmp_path):
    train, val, test = regression_splits(tmp_path)
    first = hm.run_training(train, val, test, epochs=5, seed=7, log_every=1000)
    second = hm.run_training(train, val, test, epochs=5, seed=7, log_every=1000)
    assert first["history"]["train_loss"] == second["history"]["train_loss"]


def test_run_sweep_and_table(tmp_path):
    train, val, test = regression_splits(tmp_path)
    base = dict(optimizer_name="SGD", lr=0.05, batch_size=32, epochs=3)
    rows = he.run_sweep(train, val, test, base, "lr", [0.01, 0.1])
    assert [row["label"] for row in rows] == ["0.01", "0.1"]
    table = he.sweep_table(rows, ["mse", "r2"])
    assert "| Значение |" in table and "0.01" in table
    he.plot_sweep(rows, "тест", "mse", tmp_path / "sweep.png")
    assert (tmp_path / "sweep.png").exists()


def test_sweep_over_optimizers_uses_common_learning_rate(tmp_path):
    train, val, test = regression_splits(tmp_path)
    base = dict(optimizer_name="SGD", lr=0.5, batch_size=32, epochs=2)
    rows = he.run_sweep(train, val, test, base, "optimizer_name", ["SGD", "Adam"])
    assert len(rows) == 2


@pytest.mark.skipif(not (hd.DATA_DIR / "titanic.csv").exists(), reason="titanic.csv нет в data/")
def test_titanic_end_to_end():
    train, val, test = hd.load_experiment_splits(hd.CLASSIFICATION_CONFIG)
    result = hm.run_training(train, val, test, optimizer_name="Adam", lr=0.01, epochs=30, log_every=1000)
    assert result["metrics"]["accuracy"] > 0.68                 # лучше, чем «всегда не выжил» (около 0.62)