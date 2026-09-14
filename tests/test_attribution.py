import numpy as np
from fly_drone.attribution import CHANNELS, Attributor


def small_actor(rng, n=12, hidden=6):
    return {
        "mean": rng.uniform(0, 0.5, n).tolist(),
        "scale": rng.uniform(0.1, 0.4, n).tolist(),
        "layers": [
            {
                "weights": rng.normal(0, 0.5, (hidden, n)).tolist(),
                "bias": rng.normal(0, 0.1, hidden).tolist(),
            },
            {
                "weights": rng.normal(0, 0.3, (4, hidden)).tolist(),
                "bias": rng.normal(0, 0.05, 4).tolist(),
            },
        ],
        "action_limits": [0.7, 0.5, 0.3, 0.8],
    }


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    attributor = Attributor(small_actor(rng), [f"T{i % 3}" for i in range(12)])
    f = rng.uniform(0, 1, 12)
    command, jac = attributor.forward(f)
    eps = 1e-6
    for j in range(12):
        step = np.zeros(12)
        step[j] = eps
        numeric = (
            attributor.forward(f + step)[0] - attributor.forward(f - step)[0]
        ) / (2 * eps)
        np.testing.assert_allclose(jac[:, j], numeric, atol=1e-6)
    assert np.all(np.abs(command) <= attributor.limits + 1e-12)


def test_type_totals_sum_cell_contributions_and_rank_the_driver():
    n = 6
    actor = {
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "layers": [
            {
                "weights": [[0, 0, 0, 0, 0, 0]] * 3 + [[0, 0, 0, 0.5, 0, 0]],
                "bias": [0.0] * 4,
            }
        ],
        "action_limits": [0.7, 0.5, 0.3, 0.8],
    }
    types = ["DNa01", "DNa01", "DNp01", "DNa02", "MN", "MN"]
    attributor = Attributor(actor, types, feature_ids=[10, 11, 12, 13, 14, 15])
    f = np.array([0.2, 0.2, 0.9, 0.6, 0.1, 0.1])
    explained = attributor.explain(f, top=2)
    yaw = explained["channels"]["yaw"]
    assert yaw["types"][0] == {"type": "DNa02", "value": 0.6 * 0.5 * 0.8}
    assert yaw["cells"][0] == 13
    assert set(explained["channels"]) == set(CHANNELS)
    _, contrib = attributor.contributions(f)
    total = sum(
        t["value"] for t in attributor.explain(f, top=10)["channels"]["yaw"]["types"]
    )
    np.testing.assert_allclose(total, contrib[3].sum())
