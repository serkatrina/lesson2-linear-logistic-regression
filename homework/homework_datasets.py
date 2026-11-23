"""Домашнее задание 2, задание 2: кастомный Dataset для CSV и эксперименты с реальными данными.

Что умеет CSVDataset:
  * загружает данные из CSV-файла;
  * числовые признаки стандартизирует (среднее 0, разброс 1), пропуски заполняет медианой;
  * бинарные признаки (Yes/No, male/female, 0/1) кодирует как 0/1;
  * категориальные признаки кодирует one-hot, пропуски заполняет самым частым значением;
  * все параметры предобработки считаются ТОЛЬКО по train и применяются к val и test.

Запуск экспериментов на реальных данных:  python homework_datasets.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VALID_TASKS = ("regression", "binary", "multiclass")


def _sorted_unique(series):
    """Уникальные значения в стабильном порядке (числа по возрастанию, строки по алфавиту)."""
    values = [v.item() if hasattr(v, "item") else v for v in series.dropna().unique()]
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def read_table(data, sep=","):
    """Принимает путь к CSV или готовый DataFrame и возвращает DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path, sep=sep)


# ---------------------------------------------------------------------------
# Предобработка признаков
# ---------------------------------------------------------------------------
class TabularPreprocessor:
    """Превращает таблицу pandas в числовую матрицу.

    Тип каждого столбца определяется автоматически:
      * bool или ровно 2 разных значения -> бинарный (0/1);
      * число с >2 значениями            -> числовой (стандартизация);
      * текст с >2 значениями            -> категориальный (one-hot).
    Автоопределение можно переопределить списками numeric_columns, categorical_columns,
    binary_columns (например, коды категорий, записанные числами).
    """

    def __init__(self, numeric_columns=None, categorical_columns=None, binary_columns=None,
                 normalize=True):
        self.forced = {}
        for kind, columns in (("numeric", numeric_columns), ("categorical", categorical_columns),
                              ("binary", binary_columns)):
            for column in columns or []:
                if column in self.forced and self.forced[column] != kind:
                    raise ValueError(f"Столбец {column!r} указан в двух списках")
                self.forced[column] = kind
        self.normalize = normalize
        self.columns_ = None
        self.specs_ = {}
        self.feature_names_ = []
        self.feature_kinds_ = []

    def _detect_kind(self, name, series):
        if name in self.forced:
            return self.forced[name]
        non_null = series.dropna()
        if non_null.empty:
            raise ValueError(f"Столбец {name!r} целиком состоит из пропусков")
        n_unique = non_null.nunique()
        if pd.api.types.is_bool_dtype(series):
            return "binary"
        if pd.api.types.is_numeric_dtype(series):
            return "binary" if n_unique <= 2 else "numeric"
        return "binary" if n_unique <= 2 else "categorical"

    def fit(self, df):
        """Запоминает параметры предобработки по обучающим данным."""
        self.columns_ = list(df.columns)
        self.specs_ = {}
        for name in self.columns_:
            series = df[name]
            kind = self._detect_kind(name, series)
            if kind == "numeric":
                values = pd.to_numeric(series, errors="coerce")
                if values.notna().sum() == 0:
                    raise ValueError(f"В столбце {name!r} нет ни одного числа")
                median = float(values.median())
                filled = values.fillna(median)
                std = float(filled.std(ddof=0))
                if not std > 0:                              # постоянный столбец: делить на 0 нельзя
                    std = 1.0
                self.specs_[name] = {"kind": kind, "median": median, "mean": float(filled.mean()), "std": std}
            elif kind == "binary":
                levels = _sorted_unique(series)
                if len(levels) > 2:
                    raise ValueError(f"Столбец {name!r} не бинарный: {len(levels)} значений")
                mapping = {level: i for i, level in enumerate(levels)}
                mode = float(series.map(mapping).mode().iloc[0])
                self.specs_[name] = {"kind": kind, "mapping": mapping, "mode": mode}
            else:
                categories = _sorted_unique(series)
                self.specs_[name] = {"kind": kind, "categories": categories, "mode": series.mode().iloc[0]}
        self.transform(df)                                   # заполняет feature_names_ и feature_kinds_
        return self

    def transform(self, df):
        """Превращает таблицу в матрицу float32, используя параметры, найденные в fit()."""
        if self.columns_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        missing = [c for c in self.columns_ if c not in df.columns]
        if missing:
            raise KeyError(f"В таблице нет столбцов: {missing}")
        blocks, names, kinds = [], [], []
        for name in self.columns_:
            spec = self.specs_[name]
            series = df[name]
            if spec["kind"] == "numeric":
                values = pd.to_numeric(series, errors="coerce").fillna(spec["median"])
                if self.normalize:
                    values = (values - spec["mean"]) / spec["std"]
                blocks.append(values.to_numpy(dtype=np.float32).reshape(-1, 1))
                names.append(name)
                kinds.append("numeric")
            elif spec["kind"] == "binary":
                mapped = series.map(spec["mapping"]).fillna(spec["mode"])
                blocks.append(mapped.to_numpy(dtype=np.float32).reshape(-1, 1))
                names.append(name)
                kinds.append("binary")
            else:
                filled = series.fillna(spec["mode"])
                for category in spec["categories"]:        # неизвестная категория даст строку из нулей
                    blocks.append((filled == category).to_numpy(dtype=np.float32).reshape(-1, 1))
                    names.append(f"{name}={category}")
                    kinds.append("categorical")
        self.feature_names_ = names
        self.feature_kinds_ = kinds
        return np.hstack(blocks).astype(np.float32)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CSVDataset(Dataset):
    """Dataset для табличных данных из CSV (или DataFrame).

    task:
      'regression'  -> y имеет форму (n, 1), float32 (по умолчанию стандартизируется);
      'binary'      -> y имеет форму (n, 1), float32, значения 0/1;
      'multiclass'  -> y имеет форму (n,), int64, значения 0..k-1.

    Чтобы val и test обрабатывались так же, как train, передайте reference=train_dataset:
    тогда используются параметры предобработки, найденные по train.
    """

    def __init__(self, data, target_column, task="regression", drop_columns=(), numeric_columns=None,
                 categorical_columns=None, binary_columns=None, normalize=True,
                 normalize_target=True, sep=",", reference=None):
        if task not in VALID_TASKS:
            raise ValueError(f"task должен быть одним из {VALID_TASKS}, а получено {task!r}")
        df = read_table(data, sep)
        if target_column not in df.columns:
            raise KeyError(f"В таблице нет целевого столбца {target_column!r}. Есть: {list(df.columns)}")
        absent = [c for c in drop_columns if c not in df.columns]
        if absent:
            raise KeyError(f"Нельзя удалить несуществующие столбцы: {absent}")
        if reference is not None and reference.task != task:
            raise ValueError("task не совпадает с task у reference-датасета")

        rows_before = len(df)
        df = df.dropna(subset=[target_column])
        if len(df) < rows_before:
            logger.warning("Удалено строк с пропуском в целевой переменной: %d", rows_before - len(df))
        if len(df) == 0:
            raise ValueError("В таблице не осталось строк")

        self.task = task
        self.target_column = target_column
        self.index = df.index.to_numpy()                      # исходные номера строк
        features = df.drop(columns=[target_column, *drop_columns])
        target = df[target_column]

        # --- признаки ---
        if reference is None:
            self.preprocessor = TabularPreprocessor(numeric_columns, categorical_columns,
                                                    binary_columns, normalize).fit(features)
        else:
            self.preprocessor = reference.preprocessor
        X = self.preprocessor.transform(features)
        self.feature_names = list(self.preprocessor.feature_names_)
        self.feature_kinds = list(self.preprocessor.feature_kinds_)

        # --- целевая переменная ---
        self.target_mean, self.target_std = 0.0, 1.0
        self.class_labels = None
        if task == "regression":
            values = pd.to_numeric(target, errors="coerce")
            if values.isna().any():
                raise ValueError("Целевая переменная содержит нечисловые значения")
            if reference is not None:
                self.target_mean, self.target_std = reference.target_mean, reference.target_std
            elif normalize_target:
                std = float(values.std(ddof=0))
                self.target_mean, self.target_std = float(values.mean()), std if std > 0 else 1.0
            y = ((values - self.target_mean) / self.target_std).to_numpy(dtype=np.float32).reshape(-1, 1)
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            if reference is None:
                labels = _sorted_unique(target)
                if task == "binary" and len(labels) != 2:
                    raise ValueError(f"Для binary нужно ровно 2 класса, найдено {len(labels)}. "
                                     "Для большего числа классов используйте task='multiclass'.")
                if len(labels) < 2:
                    raise ValueError("Для классификации нужно минимум 2 класса")
                self.class_labels = labels
            else:
                self.class_labels = reference.class_labels
            codes = target.map({label: i for i, label in enumerate(self.class_labels)})
            if codes.isna().any():
                raise ValueError("В целевой переменной есть метки, которых не было в train")
            if task == "binary":
                y_tensor = torch.tensor(codes.to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
            else:
                y_tensor = torch.tensor(codes.to_numpy(dtype=np.int64), dtype=torch.long)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y_tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def input_dim(self):
        return self.X.shape[1]

    @property
    def num_classes(self):
        if self.task == "regression":
            return None
        return len(self.class_labels)

    @property
    def numeric_feature_indices(self):
        """Номера столбцов X, которые являются числовыми (стандартизированными) признаками."""
        return [i for i, kind in enumerate(self.feature_kinds) if kind == "numeric"]

    def inverse_target(self, y):
        """Возвращает предсказания регрессии в исходные единицы измерения."""
        return y * self.target_std + self.target_mean

    def with_features(self, X_new, feature_names=None):
        """Копия датасета с другой матрицей признаков (для feature engineering)."""
        import copy
        clone = copy.copy(self)
        clone.X = X_new
        clone.feature_names = list(feature_names) if feature_names is not None else \
            [f"f{i}" for i in range(X_new.shape[1])]
        clone.feature_kinds = ["numeric"] * X_new.shape[1]
        return clone

    def summary(self):
        text = (f"CSVDataset(task={self.task}, объектов={len(self)}, признаков={self.input_dim}, "
                f"целевой столбец={self.target_column!r})")
        if self.class_labels is not None:
            text += f", классы={self.class_labels}"
        return text


# ---------------------------------------------------------------------------
# Разбиение на train / val / test
# ---------------------------------------------------------------------------
def split_indices(n, val_size, test_size, seed, labels=None):
    """Случайное разбиение номеров строк. Если заданы labels, доли классов сохраняются (stratify)."""
    if not (0 <= val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("Нужно 0 <= val_size, 0 < test_size и val_size + test_size < 1")
    rng = np.random.default_rng(seed)
    if labels is None:
        groups = [np.arange(n)]
    else:
        labels = np.asarray(labels)
        groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    train, val, test = [], [], []
    for group in groups:
        group = rng.permutation(group)
        n_test = int(round(len(group) * test_size))
        n_val = int(round(len(group) * val_size))
        test.append(group[:n_test])
        val.append(group[n_test:n_test + n_val])
        train.append(group[n_test + n_val:])
    return tuple(rng.permutation(np.concatenate(part)) for part in (train, val, test))


def load_csv_splits(data, target_column, task="regression", val_size=0.15, test_size=0.15, seed=42,
                    sep=",", **dataset_kwargs):
    """Читает CSV и возвращает три датасета (train, val, test).

    Параметры предобработки подбираются по train и применяются к val и test.
    """
    df = read_table(data, sep)
    if target_column not in df.columns:
        raise KeyError(f"В таблице нет целевого столбца {target_column!r}")
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    labels = df[target_column].to_numpy() if task != "regression" else None
    train_idx, val_idx, test_idx = split_indices(len(df), val_size, test_size, seed, labels)
    train = CSVDataset(df.iloc[train_idx], target_column, task, **dataset_kwargs)
    val = CSVDataset(df.iloc[val_idx], target_column, task, reference=train, **dataset_kwargs)
    test = CSVDataset(df.iloc[test_idx], target_column, task, reference=train, **dataset_kwargs)
    return train, val, test


def find_dataset(name):
    """Ищет CSV в папке data/ (затем рядом со скриптом)."""
    for folder in (DATA_DIR, BASE_DIR):
        path = folder / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Файл {name} не найден ни в {DATA_DIR}, ни в {BASE_DIR}. "
                            "Положите его в папку data/.")


# ---------------------------------------------------------------------------
# 2.2 Эксперименты на реальных датасетах
# ---------------------------------------------------------------------------
REGRESSION_CONFIG = dict(file="Student_Performance.csv", target_column="Performance Index",
                         task="regression")
CLASSIFICATION_CONFIG = dict(file="titanic.csv", target_column="Survived", task="binary",
                             drop_columns=["PassengerId", "Name", "Ticket", "Cabin"])


def load_experiment_splits(config, seed=42):
    """Загружает train/val/test для одного из описанных выше датасетов."""
    options = {k: v for k, v in config.items() if k not in ("file", "target_column", "task")}
    return load_csv_splits(find_dataset(config["file"]), config["target_column"], config["task"],
                           seed=seed, **options)


def main():
    from homework_model_modification import (compute_metrics, confusion_matrix, plot_confusion_matrix,
                                             plot_history, predict, predict_proba, run_training,
                                             save_model, setup_logging, PLOTS_DIR)
    setup_logging()

    logger.info("=== Регрессия: Student Performance ===")
    train, val, test = load_experiment_splits(REGRESSION_CONFIG)
    logger.info("%s | train=%d, val=%d, test=%d", train.summary(), len(train), len(val), len(test))
    logger.info("Признаки: %s", train.feature_names)
    result = run_training(train, val, test, optimizer_name="Adam", lr=0.01, batch_size=64,
                          epochs=100, l2=1e-4, patience=10, log_every=10)
    metrics = result["metrics"]
    baseline_mse = float(((test.y - train.y.mean()) ** 2).mean()) * train.target_std ** 2
    logger.info("Test: MSE=%.3f, RMSE=%.3f, R²=%.4f (прогноз средним даёт MSE=%.3f)",
                metrics["mse"], metrics["rmse"], metrics["r2"], baseline_mse)
    plot_history(result["history"], "Student Performance: линейная регрессия",
                 PLOTS_DIR / "student_regression_history.png")
    save_model(result["model"], "linreg_student.pth")

    logger.info("=== Бинарная классификация: Titanic ===")
    train, val, test = load_experiment_splits(CLASSIFICATION_CONFIG)
    logger.info("%s | train=%d, val=%d, test=%d", train.summary(), len(train), len(val), len(test))
    logger.info("Признаки: %s", train.feature_names)
    result = run_training(train, val, test, optimizer_name="Adam", lr=0.01, batch_size=64,
                          epochs=150, patience=15, log_every=25)
    metrics = result["metrics"]
    logger.info("Test: %s", {k: round(v, 4) for k, v in metrics.items()})
    model = result["model"]
    cm = confusion_matrix(test.y, predict(model, test.X), 2)
    plot_confusion_matrix(cm, ["не выжил", "выжил"], "Titanic: confusion matrix",
                          PLOTS_DIR / "titanic_confusion_matrix.png")
    plot_history(result["history"], "Titanic: логистическая регрессия",
                 PLOTS_DIR / "titanic_classification_history.png")
    save_model(model, "logreg_titanic.pth")


if __name__ == "__main__":
    main()