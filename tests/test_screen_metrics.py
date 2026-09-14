from fly_drone.distill import near_dodge_rates


def throw(side, hit, min_distance):
    dodged = not hit and min_distance < 2.0
    return {"side": side, "hit": hit, "min_distance": min_distance, "dodged": dodged}


def test_screen_dodge_rate_ignores_throws_that_never_came_close():
    runs = [
        {
            "threat_log": [
                throw(1.0, False, 0.6),
                throw(1.0, True, 0.3),
                throw(1.0, False, 3.5),  # never within range: not scored
                throw(-1.0, False, 0.8),
                throw(-1.0, False, 2.4),  # never within range: not scored
            ]
        }
    ]
    rates = near_dodge_rates(runs)
    assert rates["near_threats"] == 3
    assert abs(rates["near_dodge_rate"] - 2 / 3) < 1e-9
    assert rates["near_dodge_by_side"] == {"left": 0.5, "right": 1.0}
    assert rates["balanced_dodge_rate"] == 0.5


def test_screen_dodge_rate_is_unbalanced_when_a_side_never_saw_a_throw():
    rates = near_dodge_rates([{"threat_log": [throw(1.0, False, 0.5)]}])
    assert rates["near_dodge_rate"] == 1.0 and rates["balanced_dodge_rate"] is None
    assert near_dodge_rates([{"threat_log": []}])["near_dodge_rate"] is None
