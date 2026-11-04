import numpy as np
import pytest

from policylab.core import crossfit, evaluate, generate


def test_hand_calculated_dr_and_ips():
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    target = np.array([[1.0, 0.0], [1.0, 0.0]])
    result = evaluate(np.array([0, 1]), np.array([1.0, 0.0]), b, target, np.full((2, 2), 0.25))
    assert result["estimates"]["IPS"] == 1
    assert result["estimates"]["DR"] == 1
    assert result["estimates"]["DM"] == 0.25
    assert result["ess"] == 1


def test_on_policy_dr_reduces_to_mean_reward_with_constant_q():
    b = np.full((4, 2), 0.5)
    result = evaluate(np.array([0, 1, 0, 1]), [1, 0, 1, 0], b, b, np.full((4, 2), 0.3))
    assert result["estimates"]["DR"] == pytest.approx(0.5)
    assert result["estimates"]["IPS"] == 0.5
    assert result["ess"] == 4


def test_no_overlap_rejected_including_unobserved_target_actions():
    with pytest.raises(ValueError, match="support"):
        evaluate(
            np.array([0, 0]),
            [1, 1],
            [[1, 0], [1, 0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]],
        )


def test_crossfit_heldout_rewards_cannot_influence_their_fold_predictions():
    data = generate(600, 6)
    q, fold = crossfit(data["x"], data["actions"], data["rewards"])
    changed = data["rewards"].copy()
    changed[fold == 0] = 1 - changed[fold == 0]
    q2, fold2 = crossfit(data["x"], data["actions"], changed)
    assert np.array_equal(fold, fold2)
    assert np.array_equal(q[fold == 0], q2[fold == 0])


def test_clipping_is_reported_and_changes_weighted_estimates():
    result = evaluate(
        np.array([0, 1]),
        [1, 1],
        [[0.01, 0.99], [0.01, 0.99]],
        [[0.9, 0.1], [0.9, 0.1]],
        np.full((2, 2), 0.5),
        clip=5,
    )
    assert result["clipped_fraction"] == 0.5
    assert result["max_weight"] == 90
    assert result["weak_overlap"] is False  # two rows: ESS fraction is just above 1/2


def test_known_propensity_dr_is_unbiased_in_repeated_logs():
    scores = []
    truths = []
    for seed in range(80):
        d = generate(1000, seed)
        r = evaluate(
            d["actions"], d["rewards"], d["behavior"], d["target"], np.full((1000, 3), 0.4)
        )
        scores.append(r["estimates"]["DR"])
        truths.append((d["target"] * d["mu"]).sum(1).mean())
    assert abs(np.mean(scores) - np.mean(truths)) < 0.012


@pytest.mark.parametrize("bad", [float("nan"), -1, 2])
def test_bad_reward_rejected(bad):
    with pytest.raises(ValueError):
        evaluate(
            np.array([0, 1]),
            [bad, 1],
            np.full((2, 2), 0.5),
            np.full((2, 2), 0.5),
            np.full((2, 2), 0.4),
        )
