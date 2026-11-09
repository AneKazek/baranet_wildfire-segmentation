import torch

from baranet.metrics import SegmentationMetrics, boundary_f1, select_threshold


def test_macro_micro_and_zero_positive_handling():
    metrics = SegmentationMetrics(threshold=0.5)
    ones = torch.ones(2, 1, 2, 2)
    zeros = torch.zeros(2, 1, 2, 2)

    prob = torch.stack([ones[0], zeros[0]])
    target = torch.stack([ones[0], zeros[0]])
    metrics.update(prob, target, fire_id=["fire_a", "fire_b"])  # 1 positive fire
    metrics.update(zeros[:1], zeros[:1], fire_id=["fire_c"])  # all negative

    result = metrics.compute()
    assert result["n_fires"] == 1
    assert result["macro_f1"] == 1.0
    assert result["zero_positive_windows"] == 2
    # micro: TP=4, FP=0, FN=0 -> perfect; all-negative windows add no error
    assert result["micro_f1"] == 1.0


def test_false_positives_hurt_micro_not_macro():
    metrics = SegmentationMetrics()
    target = torch.zeros(2, 1, 2, 2)
    prob = torch.ones(2, 1, 2, 2)
    metrics.update(prob, target, fire_id=["a", "b"])
    result = metrics.compute()
    assert result["n_fires"] == 0
    assert result["macro_f1"] == 0.0
    assert result["micro_f1"] == 0.0
    assert result["micro_precision"] == 0.0


def test_select_threshold_prefers_higher_on_tie():
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    samples = [(target.clone(), target, None, ["a", "b"])]
    assert select_threshold(samples) == 0.95


def test_boundary_f1_identical_masks():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1.0
    result = boundary_f1(target, target, tolerance=1)
    assert result["boundary_f1"] == 1.0
    assert result["tolerance_px"] == 1.0
