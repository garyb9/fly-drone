"""Which input population can drive the loom / escape circuitry?

For every cell type with enough cells, inject a uniform current into that type
on the left side and measure the ipsilateral LC4 / LPLC2 activity, the escape
readout and the DN feature mean. This locates the injection site for a spatial
(and eventually retinotopic) encoder.
"""

import collections

import numpy as np
from fly_drone.brain import BrainRuntime

TICKS = 120
CANDIDATES = (
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "Mi1",
    "Mi4",
    "Mi9",
    "Tm1",
    "Tm2",
    "Tm3",
    "Tm4",
    "Tm9",
    "T1",
    "T2",
    "T3",
    "T4a",
    "T4b",
    "T5a",
    "T5b",
    "Lawf1",
    "Lawf2",
    "Tlp",
    "LC4",
    "LPLC2",
)


def main():
    br = BrainRuntime()
    feature_ids = br.feature_ids
    by_type = collections.defaultdict(list)
    for i, c in enumerate(br.cells):
        if c["side"].lower() == "l":
            by_type[c["type"]].append(i)

    lc4_l = br._cells("l", ("LC4",))
    lplc2_l = br._cells("l", ("LPLC2",))
    esc = br.readouts["escape"]

    rows = []
    for typ, ids in by_type.items():
        if typ not in CANDIDATES or len(ids) < 10:
            continue
        br.reset(7)
        role = br.core.input_role("scan", list(ids))
        for _ in range(TICKS):
            br.core.inject(role, 1.6)
            br.core.step(1)
        rows.append(
            (
                typ,
                len(ids),
                float(np.mean(br.core.activity(list(ids)))),
                float(np.mean(br.core.activity(lc4_l))),
                float(np.mean(br.core.activity(lplc2_l))),
                float(br.core.readout(esc)),
                float(np.mean(br.core.activity(feature_ids))),
            )
        )

    rows.sort(key=lambda r: -(r[3] + r[4]))
    print(
        f"{'type':28s} {'n':>5s} {'self':>7s} {'lc4_l':>7s} {'lplc2_l':>8s} "
        f"{'escape':>7s} {'dn':>7s}"
    )
    for typ, n, s, lc4, lp, e, dn in rows[:30]:
        print(f"{typ:28s} {n:5d} {s:7.3f} {lc4:7.3f} {lp:8.3f} {e:7.3f} {dn:7.3f}")


if __name__ == "__main__":
    main()
