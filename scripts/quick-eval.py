import sys

import numpy as np
from fly_drone.env import ConnectomeEnv
from fly_drone.plant import LIMITS

p = sys.argv[1]
e = ConnectomeEnv()
e.brain.load_policy(p)
try:
    for seed in range(1000, 1006):
        x, _ = e.reset(seed=seed)
        initial = e.info()["bearing"]
        for _ in range(100):
            x, _, done, _, info = e.step(e.brain.infer(x) / LIMITS)
        print(
            seed,
            "initial",
            round(initial, 3),
            "final",
            round(info["bearing"], 3),
            "cmd",
            np.round(e.command, 3),
            "pos",
            np.round(e.plant.pos[0], 3),
            flush=True,
        )
finally:
    e.close()
