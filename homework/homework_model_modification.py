"""Домашнее задание 2, задание 1: модификация линейной и логистической регрессии.

Что добавлено к моделям из урока:
  * L1- и L2-регуляризация (штраф к функции потерь);
  * early stopping (остановка, когда ошибка на валидации перестаёт падать);
  * многоклассовая логистическая регрессия (softmax);
  * метрики precision, recall, F1, ROC-AUC (написаны вручную на тензорах);
  * визуализация confusion matrix.

Запуск демонстрации:  python homework_model_modification.py
"""
import copy
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # рисуем сразу в файлы, без всплывающих окон
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"


def setup_logging(level=logging.INFO, log_file=None):
    """Настраивает вывод логов в консоль (и, при желании, в файл)."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=handlers, force=True)


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------
class LinearRegressionModel(nn.Module):
    """Линейная регрессия: y = w·x + b."""

    def __init__(self, in_features):
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features должен быть не меньше 1")
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegressionModel(nn.Module):
    """Логистическая регрессия.

    num_classes == 2  -> один логит (бинарная классификация, сигмоида);
    num_classes  > 2  -> num_classes логитов (многоклассовая, softmax).
    Модель возвращает логиты, а вероятности считает predict_proba().
    """

    def __init__(self, in_features, num_classes=2):
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features должен быть не меньше 1")
        if num_classes < 2:
            raise ValueError("num_classes должен быть не меньше 2")
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        return self.linear(x)


def build_model(task, input_dim, num_classes=None):
    """Создаёт модель под задачу: 'regression', 'binary' или 'multiclass'."""
    if task == "regression":
        return LinearRegressionModel(input_dim)
    if task == "binary":
        return LogisticRegressionModel(input_dim, 2)
    if task == "multiclass":
        return LogisticRegressionModel(input_dim, num_classes)
    raise ValueError(f"Неизвестная задача: {task!r}")


def build_criterion(task, num_classes=None):
    """Функция потерь под задачу."""
    if task == "regression":
        return nn.MSELoss()
    if task == "binary":
        return nn.BCEWithLogitsLoss()
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Неизвестная задача: {task!r}")


def make_optimizer(name, parameters, lr):
    """Оптимизатор по названию: SGD, SGD+momentum, Adam, RMSprop."""
    key = name.lower()
    if key == "sgd":
        return optim.SGD(parameters, lr=lr)
    if key == "sgd+momentum":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    if key == "adam":
        return optim.Adam(parameters, lr=lr)
    if key == "rmsprop":
        return optim.RMSprop(parameters, lr=lr)
    raise ValueError(f"Неизвестный оптимизатор: {name!r}")


@torch.no_grad()
def predict_proba(model, X):
    """Вероятности классов, форма (n, число_классов). Для бинарной задачи (n, 2)."""
    model.eval()
    logits = model(X)
    if model.num_classes == 2:
        p1 = torch.sigmoid(logits).reshape(-1)
        return torch.stack([1 - p1, p1], dim=1)
    return torch.softmax(logits, dim=1)


@torch.no_grad()
def predict(model, X, threshold=0.5):
    """Предсказанные метки классов (для бинарной задачи используется порог threshold)."""
    proba = predict_proba(model, X)
    if model.num_classes == 2:
        return (proba[:, 1] >= threshold).long()
    return proba.argmax(dim=1)


# ---------------------------------------------------------------------------
# Регуляризация и early stopping
# ---------------------------------------------------------------------------
def regularization_penalty(model, l1=0.0, l2=0.0):
    """Штраф регуляризации: l1 * сумма |w| + l2 * сумма w^2.

    Штрафуются только веса, смещение (bias) не штрафуется.
    """
    if l1 < 0 or l2 < 0:
        raise ValueError("Коэффициенты регуляризации не могут быть отрицательными")
    penalty = 0.0
    for name, parameter in model.named_parameters():
        if "bias" in name:
            continue
        if l1:
            penalty = penalty + l1 * parameter.abs().sum()
        if l2:
            penalty = penalty + l2 * (parameter ** 2).sum()
    return penalty


class EarlyStopping:
    """Остановка обучения, если ошибка на валидации не улучшалась `patience` эпох подряд.

    Запоминает лучшие веса и умеет вернуть их в модель (restore).
    """

    def __init__(self, patience=10, min_delta=0.0, restore_best=True):
        if patience < 1:
            raise ValueError("patience должен быть не меньше 1")
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss, model, epoch):
        """Вызывается после каждой эпохи. Возвращает True, если пора остановиться."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model):
        """Возвращает в модель веса лучшей эпохи."""
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Цикл обучения
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(model, criterion, loader):
    """Средняя потеря модели на всех батчах загрузчика."""
    model.eval()
    total, count = 0.0, 0
    for X, y in loader:
        loss = criterion(model(X), y)
        total += loss.item() * len(X)
        count += len(X)
    return total / count


def train_model(model, criterion, optimizer, train_loader, val_loader=None, epochs=100,
                l1=0.0, l2=0.0, early_stopping=None, log_every=10):
    """Универсальный цикл обучения (для линейной и логистической регрессии).

    Возвращает историю: потери на train и val по эпохам, эпоху остановки и лучшую эпоху.
    В истории train_loss записана потеря БЕЗ штрафа регуляризации, чтобы кривые были сравнимы.
    """
    if epochs < 1:
        raise ValueError("epochs должен быть не меньше 1")
    if early_stopping is not None and val_loader is None:
        raise ValueError("Для early stopping нужна валидационная выборка")

    history = {"train_loss": [], "val_loss": [], "stopped_epoch": None, "best_epoch": None}
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for X, y in train_loader:
            optimizer.zero_grad()
            data_loss = criterion(model(X), y)
            loss = data_loss + regularization_penalty(model, l1, l2)
            loss.backward()
            optimizer.step()
            total += data_loss.item() * len(X)
            count += len(X)
        train_loss = total / count
        history["train_loss"].append(train_loss)

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate_loss(model, criterion, val_loader)
            history["val_loss"].append(val_loss)

        if epoch % log_every == 0 or epoch == 1:
            message = f"Epoch {epoch}: train_loss={train_loss:.4f}"
            if val_loss is not None:
                message += f", val_loss={val_loss:.4f}"
            logger.info(message)

        if early_stopping is not None and early_stopping.step(val_loss, model, epoch):
            history["stopped_epoch"] = epoch
            logger.info("Early stopping: остановка на эпохе %d (лучшая эпоха %d)",
                        epoch, early_stopping.best_epoch)
            break

    if early_stopping is not None:
        early_stopping.restore(model)          # возвращаем веса лучшей эпохи
        history["best_epoch"] = early_stopping.best_epoch
    return history


# ---------------------------------------------------------------------------
# Метрики (написаны вручную)
# ---------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, num_classes):
    """Матрица ошибок: строки это истинные классы, столбцы это предсказанные."""
    y_true = y_true.reshape(-1).long()
    y_pred = y_pred.reshape(-1).long()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true и y_pred должны быть одной длины")
    for name, labels in (("y_true", y_true), ("y_pred", y_pred)):
        if ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError(f"В {name} есть метки вне диапазона 0..{num_classes - 1}")
    index = y_true * num_classes + y_pred                 # номер ячейки в «развёрнутой» матрице
    return torch.bincount(index, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_metrics(y_true, y_pred, num_classes):
    """Матрица ошибок, precision, recall и F1 для каждого класса."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    cm_float = cm.float()
    true_positive = torch.diag(cm_float)
    predicted_total = cm_float.sum(dim=0)                 # сколько раз предсказали класс
    actual_total = cm_float.sum(dim=1)                    # сколько объектов класса на самом деле
    zeros = torch.zeros_like(true_positive)
    precision = torch.where(predicted_total > 0, true_positive / predicted_total.clamp(min=1.0), zeros)
    recall = torch.where(actual_total > 0, true_positive / actual_total.clamp(min=1.0), zeros)
    total = precision + recall
    f1 = torch.where(total > 0, 2 * precision * recall / total.clamp(min=1e-12), zeros)
    return cm, precision, recall, f1


def roc_auc_binary(y_true, scores):
    """ROC-AUC для двух классов: площадь под ROC-кривой (метод трапеций).

    Пороги ставятся только между РАЗНЫМИ значениями оценок, поэтому одинаковые оценки
    (ничьи) обрабатываются корректно.
    """
    y_true = y_true.reshape(-1).float()
    scores = scores.reshape(-1).float()
    if y_true.shape != scores.shape:
        raise ValueError("y_true и scores должны быть одной длины")
    n_pos = y_true.sum().item()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC не определён: в y_true только один класс")

    order = torch.argsort(scores, descending=True)         # от самой высокой оценки к низкой
    y_sorted = y_true[order]
    s_sorted = scores[order]
    tps = torch.cumsum(y_sorted, dim=0)                    # сколько положительных набрано
    fps = torch.cumsum(1.0 - y_sorted, dim=0)              # сколько ложных срабатываний
    last = len(s_sorted) - 1
    change = torch.nonzero(s_sorted[1:] != s_sorted[:-1]).reshape(-1)
    idx = torch.cat([change, torch.tensor([last])])
    tpr = torch.cat([torch.zeros(1), tps[idx] / n_pos])
    fpr = torch.cat([torch.zeros(1), fps[idx] / n_neg])
    return torch.trapezoid(tpr, fpr).item()


def roc_auc_score(y_true, y_score):
    """ROC-AUC. Для многоклассовой задачи считается «один против всех» с усреднением."""
    y_true = y_true.reshape(-1).long()
    if y_score.dim() == 1 or y_score.shape[1] == 1:
        return roc_auc_binary(y_true, y_score)
    num_classes = y_score.shape[1]
    if num_classes == 2:
        return roc_auc_binary(y_true, y_score[:, 1])
    aucs = []
    for c in range(num_classes):
        is_class = y_true == c
        if is_class.sum() == 0 or is_class.all():
            continue                                       # класс отсутствует или единственный
        aucs.append(roc_auc_binary(is_class.float(), y_score[:, c]))
    if not aucs:
        raise ValueError("ROC-AUC не определён для этих данных")
    return sum(aucs) / len(aucs)


def compute_metrics(y_true, y_proba, num_classes, y_pred=None):
    """Все метрики классификации одним словарём.

    Для бинарной задачи precision/recall/F1 считаются для класса 1,
    для многоклассовой это среднее по классам (macro).
    """
    y_true = y_true.reshape(-1).long()
    if y_pred is None:
        y_pred = y_proba.argmax(dim=1)
    y_pred = y_pred.reshape(-1).long()
    _, precision, recall, f1 = per_class_metrics(y_true, y_pred, num_classes)
    accuracy = (y_true == y_pred).float().mean().item()
    if num_classes == 2:
        p, r, f = precision[1].item(), recall[1].item(), f1[1].item()
    else:
        p, r, f = precision.mean().item(), recall.mean().item(), f1.mean().item()
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy, "precision": p, "recall": r, "f1": f, "roc_auc": auc}


@torch.no_grad()
def evaluate_regression(model, X, y, target_std=1.0):
    """MSE, RMSE и R². Если целевая переменная нормализована, MSE пересчитывается в исходные единицы."""
    model.eval()
    prediction = model(X)
    squared_error = ((prediction - y) ** 2).sum().item()
    total_variance = ((y - y.mean()) ** 2).sum().item()
    mse_value = squared_error / len(y) * target_std ** 2
    r2 = 1.0 - squared_error / total_variance if total_variance > 0 else float("nan")
    return {"mse": mse_value, "rmse": mse_value ** 0.5, "r2": r2}


@torch.no_grad()
def evaluate_classification(model, X, y, num_classes):
    """Метрики классификации на тестовых данных."""
    proba = predict_proba(model, X)
    return compute_metrics(y, proba, num_classes, y_pred=predict(model, X))


def run_training(train_ds, val_ds, test_ds, optimizer_name="Adam", lr=0.01, batch_size=64,
                 epochs=50, l1=0.0, l2=0.0, patience=None, seed=42, log_every=10):
    """Полный запуск: создать модель, обучить на train, проверить на test.

    train_ds / val_ds / test_ds это датасеты с полями X, y, task, input_dim, num_classes, target_std
    (например, CSVDataset из homework_datasets.py).
    """
    torch.manual_seed(seed)
    model = build_model(train_ds.task, train_ds.input_dim, train_ds.num_classes)
    criterion = build_criterion(train_ds.task, train_ds.num_classes)
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr)
    generator = torch.Generator().manual_seed(seed)       # одинаковое перемешивание при одном seed
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    stopper = EarlyStopping(patience) if patience else None

    started = time.time()
    history = train_model(model, criterion, optimizer, train_loader, val_loader, epochs,
                          l1=l1, l2=l2, early_stopping=stopper, log_every=log_every)
    seconds = time.time() - started

    if train_ds.task == "regression":
        metrics = evaluate_regression(model, test_ds.X, test_ds.y, train_ds.target_std)
    else:
        metrics = evaluate_classification(model, test_ds.X, test_ds.y, train_ds.num_classes)
    return {"model": model, "history": history, "metrics": metrics, "seconds": seconds}


def save_model(model, name):
    """Сохраняет веса модели в папку models/ и возвращает путь."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / name
    torch.save(model.state_dict(), path)
    logger.info("Модель сохранена: %s", path)
    return path


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names=None, title="Confusion matrix", save_path=None):
    """Рисует матрицу ошибок с числами в ячейках и сохраняет в файл."""
    values = cm.tolist()
    size = len(values)
    class_names = class_names or [str(i) for i in range(size)]
    fig, ax = plt.subplots(figsize=(1.2 * size + 3, 1.2 * size + 2.5))
    image = ax.imshow(values, cmap="Blues")
    maximum = max(max(row) for row in values) or 1
    for i in range(size):
        for j in range(size):
            color = "white" if values[i][j] > maximum / 2 else "black"
            ax.text(j, i, str(values[i][j]), ha="center", va="center", color=color)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_history(history, title, save_path):
    """Кривые потерь на train и val по эпохам."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    if history["val_loss"]:
        ax.plot(history["val_loss"], label="validation")
    if history.get("best_epoch"):
        ax.axvline(history["best_epoch"] - 1, color="gray", linestyle="--", label="лучшая эпоха")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Потеря")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Синтетические данные для демонстраций и тестов
# ---------------------------------------------------------------------------
def make_regression_data(n, n_features, n_informative, noise, seed):
    """Регрессия, где важны только первые n_informative признаков, остальные шум."""
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, n_features, generator=generator)
    true_w = torch.zeros(n_features, 1)
    true_w[:n_informative, 0] = torch.linspace(3.0, -2.0, n_informative)
    y = X @ true_w + 1.0 + noise * torch.randn(n, 1, generator=generator)
    return X, y, true_w


def make_blobs(n_per_class, num_classes, seed, spread=1.0):
    """Несколько «облаков» точек на плоскости, по одному на класс."""
    generator = torch.Generator().manual_seed(seed)
    angles = torch.arange(num_classes).float() * (2 * torch.pi / num_classes)
    centers = 3.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    parts = [centers[k] + spread * torch.randn(n_per_class, 2, generator=generator)
             for k in range(num_classes)]
    X = torch.cat(parts, dim=0)
    y = torch.arange(num_classes).repeat_interleave(n_per_class)
    order = torch.randperm(len(y), generator=generator)
    return X[order], y[order]


def make_loader(X, y, batch_size, seed=0, shuffle=True):
    """DataLoader для пары тензоров."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, generator=generator)


# ---------------------------------------------------------------------------
# Демонстрации
# ---------------------------------------------------------------------------
def demo_regularization():
    """Влияние L1 и L2 на веса: из 20 признаков важны только 3."""
    logger.info("=== Демонстрация регуляризации ===")
    X, y, _ = make_regression_data(n=120, n_features=20, n_informative=3, noise=1.0, seed=0)
    X_train, y_train, X_test, y_test = X[:60], y[:60], X[60:], y[60:]
    configs = [("без регуляризации", 0.0, 0.0), ("L1, λ=0.1", 0.1, 0.0),
               ("L2, λ=0.1", 0.0, 0.1), ("L1+L2, λ=0.05", 0.05, 0.05)]
    for name, l1, l2 in configs:
        torch.manual_seed(0)
        model = LinearRegressionModel(20)
        optimizer = optim.SGD(model.parameters(), lr=0.05)
        loader = make_loader(X_train, y_train, batch_size=60)        # весь train за один батч
        train_model(model, nn.MSELoss(), optimizer, loader, epochs=300, l1=l1, l2=l2, log_every=1000)
        weights = model.linear.weight.detach()[0]
        noise_weights = weights[3:]                                    # веса неважных признаков
        test_mse = evaluate_regression(model, X_test, y_test)["mse"]
        near_zero = (noise_weights.abs() < 0.02).sum().item()
        logger.info("%-20s test MSE=%.3f | сумма |w| шумовых признаков=%.3f | почти нулевых из 17: %d",
                    name, test_mse, noise_weights.abs().sum().item(), near_zero)


def demo_early_stopping():
    """Early stopping на данных, где легко переобучиться (мало объектов, много признаков)."""
    logger.info("=== Демонстрация early stopping ===")
    X, y, _ = make_regression_data(n=125, n_features=20, n_informative=3, noise=2.0, seed=1)
    X_train, y_train, X_val, y_val = X[:25], y[:25], X[25:], y[25:]
    results = {}
    for name, patience in (("без early stopping", None), ("early stopping (patience=20)", 20)):
        torch.manual_seed(0)
        model = LinearRegressionModel(20)
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        train_loader = make_loader(X_train, y_train, batch_size=25)
        val_loader = make_loader(X_val, y_val, batch_size=100, shuffle=False)
        stopper = EarlyStopping(patience) if patience else None
        history = train_model(model, nn.MSELoss(), optimizer, train_loader, val_loader, epochs=500,
                              early_stopping=stopper, log_every=1000)
        results[name] = history
        logger.info("%-30s эпох=%d, val loss в конце=%.3f, минимум val loss=%.3f",
                    name, len(history["val_loss"]), history["val_loss"][-1], min(history["val_loss"]))
    plot_history(results["early stopping (patience=20)"], "Early stopping",
                 PLOTS_DIR / "early_stopping.png")
    plot_history(results["без early stopping"], "Без early stopping",
                 PLOTS_DIR / "no_early_stopping.png")


def demo_multiclass():
    """Многоклассовая и бинарная логистическая регрессия, метрики и confusion matrix."""
    logger.info("=== Демонстрация логистической регрессии ===")
    for num_classes in (3, 2):
        X, y = make_blobs(n_per_class=200, num_classes=num_classes, seed=0, spread=1.3)
        split = int(0.7 * len(y))
        X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]
        task = "binary" if num_classes == 2 else "multiclass"
        torch.manual_seed(0)
        model = build_model(task, 2, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        y_for_loss = y_train.float().unsqueeze(1) if task == "binary" else y_train
        loader = make_loader(X_train, y_for_loss, batch_size=32)
        train_model(model, build_criterion(task, num_classes), optimizer, loader, epochs=100, log_every=50)
        metrics = evaluate_classification(model, X_test, y_test, num_classes)
        logger.info("%d класса(ов): %s", num_classes, {k: round(v, 4) for k, v in metrics.items()})
        cm = confusion_matrix(y_test, predict(model, X_test), num_classes)
        path = plot_confusion_matrix(cm, title=f"Confusion matrix ({num_classes} класса)",
                                     save_path=PLOTS_DIR / f"confusion_matrix_{num_classes}_classes.png")
        logger.info("Confusion matrix сохранена: %s", path)


def main():
    setup_logging()
    demo_regularization()
    demo_early_stopping()
    demo_multiclass()


if __name__ == "__main__":
    main()