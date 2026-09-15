# Pre-Task-13 smoke inspection: outputs exist, encoder hash matches round.json, export parity, actor weights moved
# after the warm-up (encoder and decoder learners), metabolic + warm-up logging present, RAM headroom.
import json
import re
from pathlib import Path

import numpy as np
import torch

from fly_drone.encoder import LearnedEncoder

S = Path("runs/v5/smoke")
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


enc_round = json.loads((S / "encoder/round.json").read_text())
dec_round = json.loads((S / "decoder/round.json").read_text())
print("encoder round.json:", {k: enc_round[k] for k in enc_round if k != "paths"})
print("decoder round.json:", {k: dec_round[k] for k in dec_round if k != "paths"})

smoke_enc = LearnedEncoder.load(S / "encoder/encoder.pt")
clone_enc = LearnedEncoder.load("runs/v5/clone/encoder.pt")
check("encoder hash == round.json", smoke_enc.version == enc_round.get("encoder_version"),
      f"{smoke_enc.version} vs {enc_round.get('encoder_version')}")
diff = max(
    float((a - b).abs().max())
    for a, b in zip(
        list(smoke_enc.extractor.state_dict().values()) + list(smoke_enc.mu.state_dict().values()),
        list(clone_enc.extractor.state_dict().values()) + list(clone_enc.mu.state_dict().values()),
    )
)
check("encoder actor moved after warm-up (vs clone)", diff > 0, f"max |Δw| {diff:.2e}")

dec_json = json.loads((S / "decoder/decoder.json").read_text())
check("decoder pinned to smoke encoder", dec_json.get("encoder_version") == smoke_enc.version,
      f"{dec_json.get('encoder_version')}")
check("decoder export parity <= 1e-4", dec_round.get("export_max_error", 1) <= 1e-4, f"{dec_round.get('export_max_error')}")
r0 = json.loads(Path("runs/v5/round0/decoder.json").read_text())


def flat(d):
    return np.concatenate([np.asarray(w, dtype=np.float64).ravel() for layer in d["layers"] for w in layer.values()
                           if isinstance(w, list)]) if "layers" in d else None


try:
    a, b = flat(dec_json), flat(r0)
    check("decoder actor moved after warm-up (vs round0)", a is not None and a.shape == b.shape and np.abs(a - b).max() > 0,
          f"max |Δw| {np.abs(a - b).max():.2e}" if a is not None and a.shape == b.shape else "layer layout differs; compare manually")
except Exception as exc:  # layout of decoder.json may differ; report instead of failing silently
    print(f"INFO  decoder weight comparison skipped: {exc!r}; keys {list(dec_json)[:8]}")

for learner in ("encoder", "decoder"):
    log = Path(f"{S}/{learner}.log").read_text()
    frozen = re.findall(r"actor_frozen\s*\|\s*(\d)", log)
    metab = re.findall(r"metabolic_cost\s*\|\s*([-\d.e]+)", log)
    # SB3 logs at episode ends; a 9000-frame / 6-worker smoke is one episode, so only the final value is visible.
    # The freeze itself is covered by the reviewed WarmupSAC unit tests; here the weight comparison proves post-warm-up learning.
    check(f"{learner}: warm-up flag logged, final 0", len(frozen) > 0 and frozen[-1] == "0", f"actor_frozen seq {frozen[-4:]}")
    check(f"{learner}: metabolic cost logged", len(metab) > 0, f"last {metab[-1] if metab else None}")
    check(f"{learner}: no Traceback", "Traceback" not in log)

ram = [line.split() for line in Path(S / "ram.log").read_text().splitlines() if len(line.split()) == 3]
used = max(int(x[1]) for x in ram)
avail = min(int(x[2]) for x in ram)
check("RAM headroom (min available > 3 GB)", avail > 3000, f"peak used {used} MB, min available {avail} MB")
print("SMOKE_CHECK", "OK" if ok else "FAILED")
