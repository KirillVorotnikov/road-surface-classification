from src.core.metrics import compute_metrics, compute_confusion_matrix


class TestComputeMetrics:
    """Metrics calculation tests."""

    def test_perfect_predictions(self):
        """All predictions are correct -> all = 1.0."""
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        names = ["a", "b", "c", "d", "e"]

        m = compute_metrics(y_true, y_pred, names)

        assert m["accuracy"] == 1.0
        assert m["balanced_accuracy"] == 1.0
        assert m["f1_macro"] == 1.0

    def test_all_wrong(self):
        """All predictions are incorrect -> accuracy = 0."""
        y_true = [0, 1, 2]
        y_pred = [1, 2, 0]
        names = ["a", "b", "c"]

        m = compute_metrics(y_true, y_pred, names)

        assert m["accuracy"] == 0.0

    def test_partial_predictions(self):
        """Part is correct -> values between 0 and 1."""
        y_true = [0, 0, 1, 1, 2]
        y_pred = [0, 1, 1, 0, 2]
        names = ["a", "b", "c"]

        m = compute_metrics(y_true, y_pred, names)

        assert 0 < m["balanced_accuracy"] < 1
        assert 0 < m["f1_macro"] < 1

    def test_f1_per_class_keys(self):
        """f1_per_class contains all classes."""
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        names = ["dry", "wet", "snow"]

        m = compute_metrics(y_true, y_pred, names)

        assert "dry" in m["f1_per_class"]
        assert "wet" in m["f1_per_class"]
        assert "snow" in m["f1_per_class"]

    def test_output_keys(self):
        """The output dict contains all expected keys."""
        m = compute_metrics([0, 1], [0, 1], ["a", "b"])

        expected_keys = {
            "accuracy", "balanced_accuracy",
            "f1_macro", "f1_weighted", "f1_per_class",
        }
        assert expected_keys == set(m.keys())

    def test_single_class(self):
        """Only one class in the data → does not fall."""
        m = compute_metrics([0, 0, 0], [0, 0, 0], ["a"])

        assert m["accuracy"] == 1.0


class TestConfusionMatrix:
    """Tests confusion matrix."""

    def test_shape(self):
        """The matrix has a regular shape."""
        cm = compute_confusion_matrix([0, 1, 2], [0, 1, 2], ["a", "b", "c"])

        assert cm.shape == (3, 3)

    def test_perfect_diagonal(self):
        """With ideal predictions - only the diagonal."""
        cm = compute_confusion_matrix([0, 1, 2], [0, 1, 2], ["a", "b", "c"])

        assert cm[0, 0] == 1
        assert cm[1, 1] == 1
        assert cm[2, 2] == 1
        assert cm.sum() == 3