"""Тесты для homework_datasets.py: предобработка таблиц, CSVDataset, разбиение на выборки."""
import numpy as np
import pandas as pd
import pytest

import homework_datasets as hd


def make_frame(n=60, seed=0):
    """Небольшая таблица со всеми типами столбцов, пропусками и целевыми переменными."""
    rng = np.random.default_rng(seed)
    a = rng.normal(10, 3, n)
    a[::10] = np.nan                                         # пропуски в числовом столбце
    flag = np.where(rng.random(n) < 0.5, "no", "yes")        # бинарный столбец
    color = rng.choice(["red", "green", "blue"], n).astype(object)
    color[3] = None                                          # пропуск в категориальном столбце
    code = rng.integers(1, 4, n)                             # три значения: числовой столбец
    filled_a = np.nan_to_num(a, nan=10.0)
    target = 2 * filled_a + rng.normal(0, 1, n)
    label = np.where(filled_a > 10, "good", "bad")
    third = np.digitize(filled_a, np.quantile(filled_a, [1 / 3, 2 / 3]))
    label3 = np.array(["low", "mid", "high"])[third]
    return pd.DataFrame({"a": a, "flag": flag, "color": color, "code": code,
                         "target": target, "label": label, "label3": label3})


FEATURES = ["a", "flag", "color", "code"]


def dtype_name(tensor):
    return str(tensor.dtype)


# ----------------------------- TabularPreprocessor -----------------------------
def test_preprocessor_detects_column_types_and_names():
    frame = make_frame()
    pre = hd.TabularPreprocessor()
    X = pre.fit_transform(frame[FEATURES])
    kinds = {name: spec["kind"] for name, spec in pre.specs_.items()}
    assert kinds == {"a": "numeric", "flag": "binary", "color": "categorical", "code": "numeric"}
    assert pre.feature_names_ == ["a", "flag", "color=blue", "color=green", "color=red", "code"]
    assert X.shape == (60, 6)
    assert X.dtype == np.float32


def test_numeric_columns_are_standardized():
    X = hd.TabularPreprocessor().fit_transform(make_frame()[FEATURES])
    assert abs(X[:, 0].mean()) < 1e-5
    assert abs(X[:, 0].std() - 1) < 1e-5


def test_missing_values_are_filled():
    frame = make_frame()
    X = hd.TabularPreprocessor().fit_transform(frame[FEATURES])
    assert not np.isnan(X).any()
    # пропуск в 'a' заменён медианой, а потом стандартизован
    filled = frame["a"].fillna(frame["a"].median())
    expected = (frame["a"].median() - filled.mean()) / filled.std(ddof=0)
    assert X[0, 0] == pytest.approx(expected, abs=1e-4)


def test_binary_mapping_is_consistent_between_fit_and_transform():
    frame = make_frame()
    pre = hd.TabularPreprocessor().fit(frame[FEATURES])
    assert pre.specs_["flag"]["mapping"] == {"no": 0, "yes": 1}
    part = frame[FEATURES].iloc[:5]
    expected = [1.0 if v == "yes" else 0.0 for v in part["flag"]]
    assert pre.transform(part)[:, 1].tolist() == expected


def test_unseen_category_gives_all_zeros():
    frame = make_frame()
    pre = hd.TabularPreprocessor().fit(frame[FEATURES])
    new = frame[FEATURES].iloc[:2].copy()
    new["color"] = "purple"
    X = pre.transform(new)
    assert X[:, 2:5].sum() == 0


def test_forced_categorical_and_binary_columns():
    frame = make_frame()
    pre = hd.TabularPreprocessor(categorical_columns=["code"])
    X = pre.fit_transform(frame[FEATURES])
    assert "code=1" in pre.feature_names_ and "code=3" in pre.feature_names_
    assert X.shape == (60, 8)
    with pytest.raises(ValueError):
        hd.TabularPreprocessor(binary_columns=["color"]).fit(frame[FEATURES])   # 3 значения, не бинарный


def test_feature_kinds_are_recorded():
    pre = hd.TabularPreprocessor()
    pre.fit(make_frame()[FEATURES])
    assert pre.feature_kinds_ == ["numeric", "binary", "categorical", "categorical", "categorical", "numeric"]


def test_preprocessor_errors():
    frame = make_frame()
    with pytest.raises(RuntimeError):
        hd.TabularPreprocessor().transform(frame[FEATURES])                      # fit не вызван
    pre = hd.TabularPreprocessor().fit(frame[FEATURES])
    with pytest.raises(KeyError):
        pre.transform(frame[["a", "flag"]])                                      # нет столбцов
    broken = frame[FEATURES].copy()
    broken["empty"] = np.nan
    with pytest.raises(ValueError):
        hd.TabularPreprocessor().fit(broken)                                     # столбец из одних NaN
    with pytest.raises(ValueError):
        hd.TabularPreprocessor(numeric_columns=["a"], categorical_columns=["a"])  # столбец в двух списках


def test_constant_numeric_column_does_not_crash():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "const": [5.0, 5.0, 5.0, 5.0]})
    X = hd.TabularPreprocessor(numeric_columns=["const"]).fit_transform(frame)
    assert not np.isnan(X).any()


# ----------------------------------- CSVDataset ----------------------------------
def write_csv(tmp_path, frame=None):
    frame = make_frame() if frame is None else frame
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_regression_dataset(tmp_path):
    path, frame = write_csv(tmp_path)
    ds = hd.CSVDataset(path, "target", "regression", drop_columns=["label", "label3"])
    assert len(ds) == 60 and ds.input_dim == 6 and ds.num_classes is None
    assert tuple(ds.y.shape) == (60, 1)
    assert dtype_name(ds.y).endswith("float32") and dtype_name(ds.X).endswith("float32")
    y = np.asarray(ds.y)
    assert abs(y.mean()) < 1e-5 and abs(y.std() - 1) < 1e-4
    restored = np.asarray(ds.inverse_target(ds.y)).reshape(-1)
    assert np.allclose(restored, frame["target"].to_numpy(), atol=1e-3)
    x_item, y_item = ds[0]
    assert tuple(x_item.shape) == (6,) and tuple(y_item.shape) == (1,)


def test_regression_without_target_normalization(tmp_path):
    path, frame = write_csv(tmp_path)
    ds = hd.CSVDataset(path, "target", "regression", drop_columns=["label", "label3"],
                       normalize_target=False)
    assert ds.target_mean == 0.0 and ds.target_std == 1.0
    assert np.allclose(np.asarray(ds.y).reshape(-1), frame["target"].to_numpy(), atol=1e-4)


def test_binary_dataset_maps_labels(tmp_path):
    path, frame = write_csv(tmp_path)
    ds = hd.CSVDataset(path, "label", "binary", drop_columns=["target", "label3"])
    assert ds.class_labels == ["bad", "good"] and ds.num_classes == 2
    assert tuple(ds.y.shape) == (60, 1)
    assert set(np.asarray(ds.y).reshape(-1).tolist()) == {0.0, 1.0}
    expected = (frame["label"] == "good").astype(float).to_numpy()
    assert np.array_equal(np.asarray(ds.y).reshape(-1), expected)


def test_multiclass_dataset(tmp_path):
    path, _ = write_csv(tmp_path)
    ds = hd.CSVDataset(path, "label3", "multiclass", drop_columns=["target", "label"])
    assert ds.num_classes == 3 and ds.class_labels == ["high", "low", "mid"]
    assert tuple(ds.y.shape) == (60,)
    assert dtype_name(ds.y).endswith("int64")


def test_rows_with_missing_target_are_dropped(tmp_path):
    frame = make_frame()
    frame.loc[0, "target"] = np.nan
    path, _ = write_csv(tmp_path, frame)
    ds = hd.CSVDataset(path, "target", "regression", drop_columns=["label", "label3"])
    assert len(ds) == 59


def test_dataset_errors(tmp_path):
    path, frame = write_csv(tmp_path)
    with pytest.raises(ValueError):
        hd.CSVDataset(path, "target", "unknown_task")
    with pytest.raises(KeyError):
        hd.CSVDataset(path, "no_such_column", "regression")
    with pytest.raises(KeyError):
        hd.CSVDataset(path, "target", "regression", drop_columns=["no_such_column"])
    with pytest.raises(ValueError):
        hd.CSVDataset(path, "label3", "binary", drop_columns=["target", "label"])      # 3 класса
    with pytest.raises(FileNotFoundError):
        hd.CSVDataset(tmp_path / "missing.csv", "target", "regression")
    with pytest.raises(ValueError):
        hd.CSVDataset(path, "label", "regression", drop_columns=["target", "label3"])  # текст в регрессии


def test_reference_dataset_uses_train_parameters():
    frame = make_frame()
    train = hd.CSVDataset(frame.iloc[:40], "target", "regression", drop_columns=["label", "label3"])
    val = hd.CSVDataset(frame.iloc[40:], "target", "regression", drop_columns=["label", "label3"],
                        reference=train)
    assert val.target_mean == train.target_mean and val.target_std == train.target_std
    assert val.preprocessor is train.preprocessor
    assert val.feature_names == train.feature_names


def test_reference_with_unseen_class_raises():
    frame = make_frame()
    first = frame.iloc[:30].copy()
    first["cls"] = ["a", "b"] * 15
    second = frame.iloc[30:].copy()
    second["cls"] = "c"
    train = hd.CSVDataset(first, "cls", "multiclass", drop_columns=["target", "label", "label3"])
    with pytest.raises(ValueError):
        hd.CSVDataset(second, "cls", "multiclass", drop_columns=["target", "label", "label3"],
                      reference=train)


def test_numeric_feature_indices():
    ds = hd.CSVDataset(make_frame(), "target", "regression", drop_columns=["label", "label3"])
    assert ds.numeric_feature_indices == [0, 5]


def test_with_features_returns_modified_copy():
    ds = hd.CSVDataset(make_frame(), "target", "regression", drop_columns=["label", "label3"])
    new_x = ds.X[:, :2]
    clone = ds.with_features(new_x, ["p", "q"])
    assert clone.input_dim == 2 and clone.feature_names == ["p", "q"]
    assert ds.input_dim == 6                                   # исходный датасет не изменился
    assert clone.target_std == ds.target_std


# ------------------------------------- разбиение -------------------------------------
def test_split_indices_sizes_and_disjoint():
    train, val, test = hd.split_indices(100, 0.15, 0.15, seed=1)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert sorted(list(train) + list(val) + list(test)) == list(range(100))


def test_split_indices_reproducible_and_seed_dependent():
    first = hd.split_indices(100, 0.15, 0.15, seed=1)
    second = hd.split_indices(100, 0.15, 0.15, seed=1)
    other = hd.split_indices(100, 0.15, 0.15, seed=2)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not np.array_equal(first[0], other[0])


def test_split_indices_stratified_keeps_class_share():
    labels = np.array([0] * 80 + [1] * 20)
    for part in hd.split_indices(100, 0.15, 0.15, seed=0, labels=labels):
        share = labels[part].mean()
        assert abs(share - 0.2) < 0.03


def test_split_indices_invalid_sizes():
    with pytest.raises(ValueError):
        hd.split_indices(100, 0.5, 0.5, seed=0)
    with pytest.raises(ValueError):
        hd.split_indices(100, 0.1, 0.0, seed=0)


def test_load_csv_splits_regression(tmp_path):
    path, _ = write_csv(tmp_path)
    train, val, test = hd.load_csv_splits(path, "target", "regression", drop_columns=["label", "label3"])
    assert (len(train), len(val), len(test)) == (42, 9, 9)
    assert set(train.index).isdisjoint(val.index) and set(train.index).isdisjoint(test.index)
    assert val.preprocessor is train.preprocessor
    assert train.input_dim == val.input_dim == test.input_dim


def test_load_csv_splits_classification_is_stratified(tmp_path):
    path, frame = write_csv(tmp_path)
    train, val, test = hd.load_csv_splits(path, "label", "binary", drop_columns=["target", "label3"])
    overall = float((frame["label"] == "good").mean())
    for ds in (train, val, test):
        assert abs(float(np.asarray(ds.y).mean()) - overall) < 0.12


def test_find_dataset_error_message():
    with pytest.raises(FileNotFoundError):
        hd.find_dataset("definitely_missing_file.csv")


@pytest.mark.skipif(not (hd.DATA_DIR / "titanic.csv").exists() or
                    not (hd.DATA_DIR / "Student_Performance.csv").exists(),
                    reason="реальные датасеты не лежат в data/")
def test_real_datasets_load():
    train, val, test = hd.load_experiment_splits(hd.REGRESSION_CONFIG)
    assert train.input_dim == 5 and len(train) + len(val) + len(test) == 10000
    train, val, test = hd.load_experiment_splits(hd.CLASSIFICATION_CONFIG)
    assert train.num_classes == 2 and train.input_dim == 9
    assert not np.isnan(np.asarray(train.X)).any()