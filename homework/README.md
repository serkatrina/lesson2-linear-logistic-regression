# Домашнее задание к уроку 2: линейная и логистическая регрессия

Решение трёх заданий из `HOMEWORK.md`. Весь код работает на CPU и не требует GPU.

## Структура

```
homework/
├── homework_model_modification.py   # задание 1: модели, регуляризация, early stopping, метрики
├── homework_datasets.py             # задание 2: CSVDataset, предобработка, обучение на реальных данных
├── homework_experiments.py          # задание 3: гиперпараметры и feature engineering
├── test_model_modification.py       # unit-тесты для задания 1
├── test_datasets.py                 # unit-тесты для задания 2
├── test_experiments.py              # тесты для задания 3 и связки с обучением
├── data/                            # датасеты: Student_Performance.csv, titanic.csv
├── models/                          # сохранённые веса моделей (создаются при запуске)
├── plots/                           # графики и confusion matrix (создаются при запуске)
├── RESULTS.md                       # таблицы экспериментов (создаётся homework_experiments.py)
└── requirements.txt
```

## Как запустить

```bash
pip install -r requirements.txt
python -m pytest -v                       # тесты
python homework_model_modification.py     # демонстрации: регуляризация, early stopping, метрики
python homework_datasets.py               # обучение на реальных датасетах
python homework_experiments.py            # эксперименты (несколько минут), создаёт RESULTS.md
```

## Что сделано

### Задание 1: модификация моделей (`homework_model_modification.py`)

* **L1 и L2 регуляризация.** `regularization_penalty` добавляет к функции потерь `l1·Σ|w| + l2·Σw²`
  (только веса, смещение не штрафуется). Параметры `l1` и `l2` есть у `train_model`.
* **Early stopping.** Класс `EarlyStopping` следит за потерей на validation, останавливает обучение
  после `patience` эпох без улучшения (с учётом `min_delta`) и возвращает веса лучшей эпохи.
* **Многоклассовая логистическая регрессия.** `LogisticRegressionModel(in_features, num_classes)`:
  при двух классах один логит и сигмоида (`BCEWithLogitsLoss`), при трёх и более классах логиты
  по числу классов и softmax (`CrossEntropyLoss`).
* **Метрики вручную:** accuracy, precision, recall, F1 (через матрицу ошибок), ROC-AUC
  (площадь под кривой методом трапеций, с корректной обработкой одинаковых оценок; для многих
  классов подход «один против всех»).
* **Confusion matrix:** `plot_confusion_matrix` рисует матрицу с числами и сохраняет в `plots/`.

### Задание 2: датасеты (`homework_datasets.py`)

* `CSVDataset` читает CSV и сам определяет тип каждого столбца: число, бинарный (в том числе
  Yes/No, male/female), категориальный. Числа стандартизируются, пропуски заполняются (медиана
  и самое частое значение), категории кодируются one-hot, бинарные признаки в 0/1.
* Параметры предобработки считаются только по train (`reference=train_dataset` для val и test),
  чтобы не было утечки данных. Разбиение на train/val/test стратифицированное для классификации.
* Регрессия: **Student Performance** (10 000 строк, цель `Performance Index`).
  Бинарная классификация: **Titanic** (цель `Survived`; есть пропуски, текстовые и категориальные
  столбцы, лишние столбцы `PassengerId`, `Name`, `Ticket`, `Cabin` удаляются).

### Задание 3: эксперименты (`homework_experiments.py`)

* **3.1.** Влияние learning rate, размера батча и оптимизатора (SGD, SGD+momentum, Adam, RMSprop)
  на обе задачи. Результаты в виде графиков (`plots/sweep_*.png`) и таблиц (`RESULTS.md`).
* **3.2.** Feature engineering: полиномиальные признаки (x²), взаимодействия (x_i·x_j),
  статистические признаки (среднее и дисперсия по строке) и их комбинация. Сравнение с базовой
  моделью, усреднение по трём запускам с разными seed.

## Результаты и выводы

Отчёт с числами, анализом и ограничениями: `REPORT.md`. Подробные таблицы экспериментов создаются
при запуске `homework_experiments.py` и сохраняются в `RESULTS.md`, графики в `plots/`.