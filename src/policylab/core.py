"""Cross-fitted off-policy evaluation with known logging propensities."""

import numpy as np


def distribution(p):
    p = np.asarray(p, float)
    if (
        p.ndim != 2
        or len(p) < 2
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
        or not np.allclose(p.sum(1), 1, atol=1e-7)
    ):
        raise ValueError("need at least two finite probability rows")
    return p


def softmax(z):
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def reward_means(x):
    z = np.column_stack(
        [
            x[:, 0] + 0.6 * x[:, 1] ** 2,
            -x[:, 0] + np.sin(x[:, 2]),
            0.3 + x[:, 1] - 0.7 * x[:, 2] ** 2,
        ]
    )
    return 1 / (1 + np.exp(-z))


def target_policy(x):
    # A fixed, context-only target; it is not learned on evaluation rewards.
    return softmax(5 * reward_means(x))


def generate(n, seed, regime="good"):
    if n < 10 or regime not in {"good", "poor"}:
        raise ValueError("need at least 10 rows and known overlap regime")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    mu = reward_means(x)
    strength, floor = (1.0, 0.12) if regime == "good" else (9.0, 0.002)
    behavior = (1 - 3 * floor) * softmax(-strength * mu) + floor
    actions = (rng.random(n)[:, None] > np.cumsum(behavior, axis=1)).sum(1)
    rewards = (rng.random(n) < mu[np.arange(n), actions]).astype(float)
    return {
        "x": x,
        "actions": actions,
        "rewards": rewards,
        "behavior": behavior,
        "target": target_policy(x),
        "mu": mu,
    }


def features(x):
    x = np.asarray(x, float)
    return np.column_stack([np.ones(len(x)), x])  # intentionally misspecified linear reward model


def crossfit(x, actions, rewards, seed=17, folds=5, ridge=1.0):
    x = np.asarray(x, float)
    a = np.asarray(actions)
    r = np.asarray(rewards, float)
    if x.ndim != 2 or len(x) < folds or folds < 2 or not np.isfinite(x).all() or ridge <= 0:
        raise ValueError("invalid cross-fit design")
    if (
        a.shape != (len(x),)
        or a.dtype.kind not in "iu"
        or np.any((a < 0) | (a > 2))
        or r.shape != (len(x),)
        or not np.isfinite(r).all()
        or np.any((r < 0) | (r > 1))
    ):
        raise ValueError("invalid logged feedback")
    fold = np.empty(len(x), int)
    fold[np.random.default_rng(seed).permutation(len(x))] = np.arange(len(x)) % folds
    z = features(x)
    q = np.zeros((len(x), 3))
    for k in range(folds):
        held = fold == k
        for arm in range(3):
            train = (~held) & (a == arm)
            if train.sum() < 2:
                raise ValueError("not enough observations for an action in a training fold")
            train_z = z[train]
            coef = np.linalg.solve(
                train_z.T @ train_z + ridge * np.eye(z.shape[1]), train_z.T @ r[train]
            )
            q[held, arm] = np.clip(z[held] @ coef, 0, 1)
    return q, fold


def evaluate(actions, rewards, behavior, target, q, clip=None):
    behavior = distribution(behavior)
    target = distribution(target)
    a = np.asarray(actions)
    r = np.asarray(rewards, float)
    q = np.asarray(q, float)
    n, k = target.shape
    if (
        behavior.shape != target.shape
        or q.shape != target.shape
        or not np.isfinite(q).all()
        or np.any((q < 0) | (q > 1))
        or a.shape != (n,)
        or a.dtype.kind not in "iu"
        or np.any((a < 0) | (a >= k))
        or r.shape != (n,)
        or not np.isfinite(r).all()
        or np.any((r < 0) | (r > 1))
    ):
        raise ValueError("invalid logged rows or reward predictions")
    if np.any((target > 0) & (behavior <= 0)):
        raise ValueError("target policy lacks logging support")
    if clip is not None and (not np.isfinite(clip) or clip <= 0):
        raise ValueError("clip must be finite and positive")
    observed = behavior[np.arange(n), a]
    if np.any(observed <= 0):
        raise ValueError("observed action cannot have zero propensity")
    w = target[np.arange(n), a] / observed
    used = w if clip is None else np.minimum(w, clip)
    direct = (target * q).sum(1)
    residual = r - q[np.arange(n), a]
    scores = {"IPS": used * r, "DM": direct, "DR": direct + used * residual}
    estimates = {name: float(v.mean()) for name, v in scores.items()}
    estimates["SNIPS"] = float((used * r).sum() / used.sum()) if used.sum() > 0 else None
    se = {name: float(v.std(ddof=1) / np.sqrt(n)) for name, v in scores.items()}
    ess = float(w.sum() ** 2 / np.square(w).sum()) if np.square(w).sum() > 0 else 0.0
    return {
        "estimates": estimates,
        "se": se,
        "ess": ess,
        "ess_fraction": ess / n,
        "max_weight": float(w.max()),
        "clipped_fraction": float((w > used).mean()),
        "weak_overlap": ess / n < 0.2,
        "clip": clip,
    }
