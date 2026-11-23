"""Домашнее задание 2, задание 3: эксперименты и анализ.

3.1  Исследование гиперпараметров: скорость обучения, размер батча, оптимизатор.
3.2  Feature engineering: полиномиальные признаки, взаимодействия, статистические признаки.

Запуск:  python homework_experiments.py
Графики сохраняются в plots/, таблицы с результатами в RESULTS.md.
"""
import logging
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from homework_datasets import CLASSIFICATION_CONFIG, REGRESSION_CONFIG, load_experiment_splits
from homework_model_modification import PLOTS_DIR, run_training, setup_logging

logger = logging.getLogger(__name__)
RESULTS_PATH = PLOTS_DIR.parent / "RESULTS.md"

# Для каждой задачи: датасет, основная метрика, настройки по умолчанию для экспериментов
TASKS = {
    "Регрессия (Student Performance)": dict(config=REGRESSION_CONFIG, metric="mse", better="меньше",
                                            base=dict(optimizer_name="SGD", lr=0.05, batch_size=64, epochs=30)),
    "Классификация (Titanic)": dict(config=CLASSIFICATION_CONFIG, metric="roc_auc", better="больше",
                                    base=dict(optimizer_name="SGD", lr=0.05, batch_size=64, epochs=60)),
}


# ---------------------------------------------------------------------------
# 3.1 Гиперпараметры
# ---------------------------------------------------------------------------
def run_sweep(train, val, test, base, parameter, values):
    """Запускает обучение для каждого значения одного гиперпараметра (остальные фиксированы)."""
    rows = []
    for value in values:
        settings = dict(base)
        settings[parameter] = value
        if parameter == "optimizer_name":
            settings["lr"] = 0.01                   # для честного сравнения оптимизаторов шаг одинаков
        logger.info("Эксперимент: %s", {k: settings[k] for k in ("optimizer_name", "lr", "batch_size")})
        result = run_training(train, val, test, seed=42, log_every=1000, **settings)
        rows.append({"label": str(value), "history": result["history"],
                     "metrics": result["metrics"], "seconds": result["seconds"]})
    return rows


def sweep_table(rows, metric_names):
    """Таблица результатов в формате Markdown."""
    header = "| Значение | val loss (в конце) | " + " | ".join(metric_names) + " | время, с |"
    lines = [header, "|" + "---|" * (len(metric_names) + 3)]
    for row in rows:
        values = " | ".join(f"{row['metrics'][m]:.4f}" for m in metric_names)
        lines.append(f"| {row['label']} | {row['history']['val_loss'][-1]:.4f} | {values} | {row['seconds']:.1f} |")
    return "\n".join(lines)


def plot_sweep(rows, title, metric, path):
    """Слева кривые val loss для каждого значения, справа итоговая метрика на test."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4))
    for row in rows:
        ax_left.plot(row["history"]["val_loss"], label=row["label"])
    ax_left.set_xlabel("Эпоха")
    ax_left.set_ylabel("Потеря на validation")
    ax_left.set_yscale("log")
    ax_left.legend()
    ax_left.grid(True)
    ax_left.set_title(title)
    labels = [row["label"] for row in rows]
    ax_right.bar(labels, [row["metrics"][metric] for row in rows])
    ax_right.set_ylabel(f"{metric} на test")
    ax_right.set_title(f"Итоговая метрика: {metric}")
    ax_right.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def hyperparameter_experiments(report):
    """Три серии экспериментов для каждой из двух задач."""
    series = [
        ("learning rate", "lr", [0.001, 0.01, 0.05, 0.1, 0.5]),
        ("размер батча", "batch_size", [16, 64, 256, 1024]),
        ("оптимизатор (lr=0.01)", "optimizer_name", ["SGD", "SGD+momentum", "Adam", "RMSprop"]),
    ]
    for task_name, spec in TASKS.items():
        train, val, test = load_experiment_splits(spec["config"])
        metric_names = ["mse", "r2"] if spec["metric"] == "mse" else ["accuracy", "f1", "roc_auc"]
        report.append(f"## 3.1 {task_name}\n")
        report.append(f"Базовые настройки: {spec['base']}. Основная метрика: `{spec['metric']}` "
                      f"({spec['better']} значение лучше).\n")
        for series_name, parameter, values in series:
            logger.info("=== %s: %s ===", task_name, series_name)
            rows = run_sweep(train, val, test, spec["base"], parameter, values)
            file_name = f"sweep_{parameter}_{'regression' if spec['metric'] == 'mse' else 'classification'}.png"
            plot_sweep(rows, f"{task_name}: {series_name}", spec["metric"], PLOTS_DIR / file_name)
            report.append(f"### {series_name}\n")
            report.append(sweep_table(rows, metric_names) + "\n")
            report.append(f"График: `plots/{file_name}`\n")


# ---------------------------------------------------------------------------
# 3.2 Feature engineering
# ---------------------------------------------------------------------------
def polynomial_features(X, names, numeric_idx, degree=2):
    """Степени x^2, ..., x^degree числовых признаков."""
    if not numeric_idx or degree < 2:
        return X[:, :0], []
    base = X[:, numeric_idx]
    blocks = [base ** power for power in range(2, degree + 1)]
    new_names = [f"{names[i]}^{power}" for power in range(2, degree + 1) for i in numeric_idx]
    return torch.cat(blocks, dim=1), new_names


def interaction_features(X, names):
    """Попарные произведения всех признаков x_i * x_j (i < j)."""
    if X.shape[1] < 2:
        return X[:, :0], []
    first, second = torch.triu_indices(X.shape[1], X.shape[1], offset=1)
    new_names = [f"{names[i]}*{names[j]}" for i, j in zip(first.tolist(), second.tolist())]
    return X[:, first] * X[:, second], new_names


def statistical_features(X, numeric_idx):
    """Среднее и дисперсия числовых признаков каждого объекта (по строке)."""
    if not numeric_idx:
        return X[:, :0], []
    base = X[:, numeric_idx]
    stats = torch.stack([base.mean(dim=1), base.var(dim=1, correction=0)], dim=1)
    return stats, ["row_mean", "row_var"]


def make_new_features(dataset, methods):
    """Склеивает новые признаки выбранных методов: 'poly', 'inter', 'stat'."""
    X, names, numeric = dataset.X, dataset.feature_names, dataset.numeric_feature_indices
    blocks, new_names = [], []
    if "poly" in methods:
        block, block_names = polynomial_features(X, names, numeric)
        blocks.append(block)
        new_names += block_names
    if "inter" in methods:
        block, block_names = interaction_features(X, names)
        blocks.append(block)
        new_names += block_names
    if "stat" in methods:
        block, block_names = statistical_features(X, numeric)
        blocks.append(block)
        new_names += block_names
    return torch.cat(blocks, dim=1), new_names


def engineer_datasets(train, val, test, methods):
    """Добавляет новые признаки к train/val/test. Новые столбцы стандартизируются по train."""
    if not methods:
        return train, val, test
    new_train, new_names = make_new_features(train, methods)
    mean = new_train.mean(dim=0)
    std = new_train.std(dim=0, correction=0).clamp(min=1e-6)
    result = []
    for dataset in (train, val, test):
        new = new_train if dataset is train else make_new_features(dataset, methods)[0]
        X_new = torch.cat([dataset.X, (new - mean) / std], dim=1)
        result.append(dataset.with_features(X_new, dataset.feature_names + new_names))
    return tuple(result)


FEATURE_METHODS = {
    "baseline (без новых признаков)": (),
    "полиномиальные (x²)": ("poly",),
    "взаимодействия (x_i·x_j)": ("inter",),
    "статистические (mean, var)": ("stat",),
    "все вместе": ("poly", "inter", "stat"),
}


def feature_engineering_experiments(report, seeds=(0, 1, 2)):
    """Сравнение моделей с новыми признаками и без них (усреднение по нескольким seed)."""
    for task_name, spec in TASKS.items():
        train, val, test = load_experiment_splits(spec["config"])
        metric_names = ["mse", "r2"] if spec["metric"] == "mse" else ["accuracy", "f1", "roc_auc"]
        logger.info("=== Feature engineering: %s ===", task_name)
        summary = {}
        for method_name, methods in FEATURE_METHODS.items():
            tr, va, te = engineer_datasets(train, val, test, methods)
            per_seed = {m: [] for m in metric_names}
            for seed in seeds:
                result = run_training(tr, va, te, optimizer_name="Adam", lr=0.01, batch_size=64,
                                      epochs=100, patience=10, seed=seed, log_every=1000)
                for m in metric_names:
                    per_seed[m].append(result["metrics"][m])
            summary[method_name] = {"features": tr.input_dim, "per_seed": per_seed}
            logger.info("%-34s признаков=%d, %s=%.4f", method_name, tr.input_dim, spec["metric"],
                        statistics.mean(per_seed[spec["metric"]]))

        report.append(f"## 3.2 Feature engineering: {task_name}\n")
        report.append(f"Среднее ± стандартное отклонение по {len(seeds)} запускам (разные seed). "
                      f"Adam, lr=0.01, batch=64, early stopping (patience=10). "
                      f"Основная метрика `{spec['metric']}`: {spec['better']} значение лучше.\n")
        header = "| Набор признаков | Число признаков | " + " | ".join(metric_names) + " |"
        report.append(header)
        report.append("|" + "---|" * (len(metric_names) + 2))
        for method_name, info in summary.items():
            cells = " | ".join(f"{statistics.mean(info['per_seed'][m]):.4f} ± "
                               f"{statistics.pstdev(info['per_seed'][m]):.4f}" for m in metric_names)
            report.append(f"| {method_name} | {info['features']} | {cells} |")
        report.append("")

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = list(summary.keys())
        means = [statistics.mean(summary[k]["per_seed"][spec["metric"]]) for k in labels]
        errors = [statistics.pstdev(summary[k]["per_seed"][spec["metric"]]) for k in labels]
        ax.barh(labels, means, xerr=errors)
        ax.set_xlabel(f"{spec['metric']} на test ({spec['better']} значение лучше)")
        ax.set_title(f"Feature engineering: {task_name}")
        fig.tight_layout()
        file_name = f"feature_engineering_{'regression' if spec['metric'] == 'mse' else 'classification'}.png"
        fig.savefig(PLOTS_DIR / file_name, dpi=120)
        plt.close(fig)
        report.append(f"График: `plots/{file_name}`\n")


def main():
    setup_logging()
    PLOTS_DIR.mkdir(exist_ok=True)
    report = ["# Результаты экспериментов\n",
              "Файл создаётся автоматически скриптом `homework_experiments.py`.\n"]
    hyperparameter_experiments(report)
    feature_engineering_experiments(report)
    RESULTS_PATH.write_text("\n".join(report), encoding="utf-8")
    logger.info("Результаты записаны в %s", RESULTS_PATH)


if __name__ == "__main__":
    main()