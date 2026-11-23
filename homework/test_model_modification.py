"""Тесты для homework_model_modification.py: регуляризация, early stopping, метрики, обучение."""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

import homework_model_modification as hm


def fixed_linear_model(weights, bias):
    """Линейная модель с заданными весами (чтобы результаты можно было посчитать вручную)."""
    model = hm.LinearRegressionModel(len(weights))
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([weights]))
        model.linear.bias.fill_(bias)
    return model


# ------------------------------- регуляризация -------------------------------
def test_regularization_penalty_values():
    model = fixed_linear_model([1.0, -2.0], 5.0)
    # L1: 0.1 * (|1| + |-2|) = 0.3;  L2: 0.01 * (1 + 4) = 0.05;  bias не штрафуется
    assert hm.regularization_penalty(model, l1=0.1, l2=0.01).item() == pytest.approx(0.35)
    assert hm.regularization_penalty(model, l1=0.1).item() == pytest.approx(0.3)
    assert hm.regularization_penalty(model, l2=0.01).item() == pytest.approx(0.05)
    assert hm.regularization_penalty(model) == 0.0                # без штрафа возвращается обычный 0.0


def test_regularization_penalty_gradients():
    model = fixed_linear_model([1.0, -2.0], 5.0)
    hm.regularization_penalty(model, l2=0.5).backward()
    assert torch.allclose(model.linear.weight.grad, model.linear.weight.detach())   # 2 * 0.5 * w = w
    assert model.linear.bias.grad is None                                           # bias не штрафуется
    model.zero_grad()
    hm.regularization_penalty(model, l1=1.0).backward()
    assert torch.equal(model.linear.weight.grad, torch.tensor([[1.0, -1.0]]))       # sign(w)


def test_regularization_negative_coefficients_raise():
    model = fixed_linear_model([1.0], 0.0)
    with pytest.raises(ValueError):
        hm.regularization_penalty(model, l1=-0.1)
    with pytest.raises(ValueError):
        hm.regularization_penalty(model, l2=-0.1)


def weight_norm_after_training(l1=0.0, l2=0.0):
    X, y, _ = hm.make_regression_data(n=60, n_features=20, n_informative=3, noise=1.0, seed=0)
    torch.manual_seed(0)
    model = hm.LinearRegressionModel(20)
    loader = hm.make_loader(X, y, batch_size=60)
    hm.train_model(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.05), loader,
                   epochs=200, l1=l1, l2=l2, log_every=1000)
    return model.linear.weight.abs().sum().item()


def test_l1_and_l2_shrink_weights():
    plain = weight_norm_after_training()
    assert weight_norm_after_training(l1=0.5) < plain
    assert weight_norm_after_training(l2=0.5) < plain


# -------------------------------- early stopping --------------------------------
def test_early_stopping_stops_and_restores_best_weights():
    model = fixed_linear_model([1.0], 0.0)
    stopper = hm.EarlyStopping(patience=2)
    decisions = []
    for epoch, loss in enumerate([1.0, 0.9, 0.95, 0.96, 0.97], start=1):
        with torch.no_grad():
            model.linear.bias.fill_(float(epoch))             # так видно, веса какой эпохи вернулись
        decisions.append(stopper.step(loss, model, epoch))
        if decisions[-1]:
            break
    assert decisions == [False, False, False, True]
    assert stopper.best_epoch == 2
    stopper.restore(model)
    assert model.linear.bias.item() == pytest.approx(2.0)


def test_early_stopping_min_delta():
    model = fixed_linear_model([1.0], 0.0)
    stopper = hm.EarlyStopping(patience=2, min_delta=0.1)
    # улучшения меньше min_delta не считаются улучшениями
    results = [stopper.step(loss, model, epoch) for epoch, loss in enumerate([1.0, 0.95, 0.94], start=1)]
    assert results == [False, False, True]


def test_early_stopping_invalid_patience():
    with pytest.raises(ValueError):
        hm.EarlyStopping(patience=0)


def test_train_model_early_stops_when_loss_does_not_improve():
    X, y, _ = hm.make_regression_data(n=50, n_features=3, n_informative=3, noise=0.1, seed=0)
    model = hm.LinearRegressionModel(3)
    train_loader = hm.make_loader(X, y, batch_size=25)
    val_loader = hm.make_loader(X, y, batch_size=50, shuffle=False)
    optimizer = optim.SGD(model.parameters(), lr=0.0)          # шаг 0: модель не учится
    stopper = hm.EarlyStopping(patience=3)
    history = hm.train_model(model, nn.MSELoss(), optimizer, train_loader, val_loader, epochs=50,
                             early_stopping=stopper, log_every=1000)
    assert history["stopped_epoch"] == 4 and history["best_epoch"] == 1
    assert len(history["val_loss"]) == 4


def test_train_model_validation_errors():
    X, y, _ = hm.make_regression_data(n=20, n_features=2, n_informative=2, noise=0.1, seed=0)
    model = hm.LinearRegressionModel(2)
    loader = hm.make_loader(X, y, batch_size=10)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError):                           # early stopping без validation
        hm.train_model(model, nn.MSELoss(), optimizer, loader, epochs=5,
                       early_stopping=hm.EarlyStopping(2))
    with pytest.raises(ValueError):
        hm.train_model(model, nn.MSELoss(), optimizer, loader, epochs=0)


# ------------------------------------ обучение ------------------------------------
def test_linear_regression_recovers_true_weights():
    X, y, true_w = hm.make_regression_data(n=200, n_features=3, n_informative=3, noise=0.1, seed=0)
    torch.manual_seed(0)
    model = hm.LinearRegressionModel(3)
    loader = hm.make_loader(X, y, batch_size=200)             # весь train за один шаг
    history = hm.train_model(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.1), loader,
                             epochs=200, log_every=1000)
    assert history["train_loss"][-1] < history["train_loss"][0] * 0.1
    assert torch.allclose(model.linear.weight.detach().reshape(-1), true_w.reshape(-1), atol=0.05)
    assert model.linear.bias.item() == pytest.approx(1.0, abs=0.05)


def test_multiclass_logistic_regression_learns():
    X, y = hm.make_blobs(n_per_class=100, num_classes=3, seed=0, spread=0.8)
    torch.manual_seed(0)
    model = hm.build_model("multiclass", 2, 3)
    loader = hm.make_loader(X, y, batch_size=32)
    hm.train_model(model, hm.build_criterion("multiclass", 3), optim.Adam(model.parameters(), lr=0.05),
                   loader, epochs=60, log_every=1000)
    assert (hm.predict(model, X) == y).float().mean().item() > 0.95
    metrics = hm.evaluate_classification(model, X, y, 3)
    assert metrics["roc_auc"] > 0.99 and metrics["f1"] > 0.95


def test_binary_logistic_regression_learns():
    X, y = hm.make_blobs(n_per_class=100, num_classes=2, seed=0, spread=0.8)
    torch.manual_seed(0)
    model = hm.build_model("binary", 2)
    loader = hm.make_loader(X, y.float().unsqueeze(1), batch_size=32)
    hm.train_model(model, hm.build_criterion("binary"), optim.Adam(model.parameters(), lr=0.05),
                   loader, epochs=60, log_every=1000)
    assert (hm.predict(model, X) == y).float().mean().item() > 0.95


def test_logistic_model_outputs_and_predictions():
    X = torch.randn(10, 4)
    binary = hm.LogisticRegressionModel(4, 2)
    multi = hm.LogisticRegressionModel(4, 3)
    assert tuple(binary(X).shape) == (10, 1) and tuple(multi(X).shape) == (10, 3)
    for model, k in ((binary, 2), (multi, 3)):
        proba = hm.predict_proba(model, X)
        assert tuple(proba.shape) == (10, k)
        assert torch.allclose(proba.sum(dim=1), torch.ones(10), atol=1e-5)
        labels = hm.predict(model, X)
        assert labels.min() >= 0 and labels.max() < k
    with pytest.raises(ValueError):
        hm.LogisticRegressionModel(4, 1)


def test_factories():
    params = hm.LinearRegressionModel(2).parameters()
    assert isinstance(hm.make_optimizer("adam", params, 0.1), optim.Adam)
    assert isinstance(hm.make_optimizer("RMSprop", hm.LinearRegressionModel(2).parameters(), 0.1), optim.RMSprop)
    momentum = hm.make_optimizer("SGD+momentum", hm.LinearRegressionModel(2).parameters(), 0.1)
    assert momentum.defaults["momentum"] == 0.9
    with pytest.raises(ValueError):
        hm.make_optimizer("unknown", hm.LinearRegressionModel(2).parameters(), 0.1)
    assert isinstance(hm.build_criterion("regression"), nn.MSELoss)
    assert isinstance(hm.build_criterion("binary"), nn.BCEWithLogitsLoss)
    assert isinstance(hm.build_criterion("multiclass", 3), nn.CrossEntropyLoss)
    with pytest.raises(ValueError):
        hm.build_model("unknown", 2)


# ---------------------------------- метрики ----------------------------------
def test_confusion_matrix_values():
    y_true = torch.tensor([0, 1, 2, 2, 1])
    y_pred = torch.tensor([0, 2, 2, 1, 1])
    cm = hm.confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 1], [0, 1, 1]]        # строки: истина, столбцы: прогноз


def test_confusion_matrix_errors():
    with pytest.raises(ValueError):
        hm.confusion_matrix(torch.tensor([0, 3]), torch.tensor([0, 1]), 3)     # метка вне диапазона
    with pytest.raises(ValueError):
        hm.confusion_matrix(torch.tensor([0, 1]), torch.tensor([0]), 2)        # разная длина


def test_per_class_precision_recall_f1():
    y_true = torch.tensor([0, 1, 2, 2, 1])
    y_pred = torch.tensor([0, 2, 2, 1, 1])
    _, precision, recall, f1 = hm.per_class_metrics(y_true, y_pred, 3)
    assert precision.tolist() == pytest.approx([1.0, 0.5, 0.5])
    assert recall.tolist() == pytest.approx([1.0, 0.5, 0.5])
    assert f1.tolist() == pytest.approx([1.0, 0.5, 0.5])


def test_metrics_do_not_divide_by_zero():
    # класс 1 ни разу не предсказан: precision для него равен 0, а не NaN
    _, precision, recall, f1 = hm.per_class_metrics(torch.tensor([0, 1]), torch.tensor([0, 0]), 2)
    assert precision.tolist() == pytest.approx([0.5, 0.0])
    assert recall.tolist() == pytest.approx([1.0, 0.0])
    assert f1[1].item() == 0.0


def test_compute_metrics_multiclass_macro():
    y_true = torch.tensor([0, 1, 2, 2, 1])
    y_pred = torch.tensor([0, 2, 2, 1, 1])
    proba = torch.nn.functional.one_hot(y_pred, 3).float()
    metrics = hm.compute_metrics(y_true, proba, 3, y_pred=y_pred)
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["f1"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)


def test_compute_metrics_binary():
    y_true = torch.tensor([0, 0, 1, 1, 1])
    p1 = torch.tensor([0.1, 0.6, 0.8, 0.7, 0.4])
    proba = torch.stack([1 - p1, p1], dim=1)
    metrics = hm.compute_metrics(y_true, proba, 2, y_pred=(p1 >= 0.5).long())
    # TP=2, FP=1, FN=1, TN=1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["roc_auc"] == pytest.approx(5 / 6)        # 5 из 6 пар «положительный-отрицательный» упорядочены верно


def test_roc_auc_known_cases():
    assert hm.roc_auc_binary(torch.tensor([0, 0, 1, 1]), torch.tensor([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert hm.roc_auc_binary(torch.tensor([0, 1, 0, 1]), torch.tensor([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert hm.roc_auc_binary(torch.tensor([0, 0, 1, 1]), torch.tensor([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert hm.roc_auc_binary(torch.tensor([0, 0, 1, 1]), torch.tensor([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        hm.roc_auc_binary(torch.tensor([1, 1, 1]), torch.tensor([0.1, 0.2, 0.3]))   # только один класс


def test_roc_auc_multiclass_one_vs_rest():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    perfect = torch.nn.functional.one_hot(y_true, 3).float()
    assert hm.roc_auc_score(y_true, perfect) == pytest.approx(1.0)


def test_evaluate_regression_values():
    model = fixed_linear_model([2.0], 1.0)
    X = torch.tensor([[0.0], [1.0], [2.0]])
    perfect = X * 2 + 1
    result = hm.evaluate_regression(model, X, perfect)
    assert result["mse"] == pytest.approx(0.0, abs=1e-8) and result["r2"] == pytest.approx(1.0)
    noisy = perfect + torch.tensor([[1.0], [-1.0], [0.0]])       # ошибки: сумма квадратов = 2
    assert hm.evaluate_regression(model, X, noisy)["mse"] == pytest.approx(2 / 3)
    assert hm.evaluate_regression(model, X, noisy, target_std=10.0)["mse"] == pytest.approx(200 / 3)
    assert hm.evaluate_regression(model, X, noisy)["r2"] == pytest.approx(1 - 2 / 6)


# --------------------------------- визуализация ---------------------------------
def test_plot_confusion_matrix_creates_file(tmp_path):
    path = hm.plot_confusion_matrix(torch.tensor([[5, 1], [2, 7]]), ["a", "b"], save_path=tmp_path / "cm.png")
    assert path.exists() and path.stat().st_size > 0


def test_plot_history_creates_file(tmp_path):
    history = {"train_loss": [3.0, 2.0, 1.0], "val_loss": [3.5, 2.5, 2.0], "best_epoch": 3}
    hm.plot_history(history, "test", tmp_path / "history.png")
    assert (tmp_path / "history.png").exists()


def test_state_dict_roundtrip(tmp_path):
    model = fixed_linear_model([1.0, 2.0], 3.0)
    path = tmp_path / "m.pth"
    torch.save(model.state_dict(), path)
    clone = hm.LinearRegressionModel(2)
    clone.load_state_dict(torch.load(path))
    assert torch.equal(clone.linear.weight, model.linear.weight)