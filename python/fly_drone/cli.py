import argparse
import json
import sys
import time
from pathlib import Path

from .env import TASKS


def main():
    parser = argparse.ArgumentParser(description="Fly connectome drone laboratory")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--policy")
    p.add_argument("--looming-policy")
    p.add_argument(
        "--task-policy",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="policy for another task, e.g. approach=runs/x/actor.json",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--accepted",
        action="store_true",
        help="load accepted policies from docs/results/accepted-policies.json",
    )
    p.add_argument(
        "--current",
        action="store_true",
        help="load best-available (not necessarily accepted) policies from "
        "docs/results/current-policies.json",
    )
    p = sub.add_parser("assay")
    p.add_argument("--output", default="runs/sensory-assay.json")
    p = sub.add_parser("baseline")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--output", default="runs/baseline.json")
    p = sub.add_parser("train")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--task", choices=["hover", *TASKS], default="visual")
    p.add_argument("--output", default="runs/visual")
    p.add_argument("--resume")
    p.add_argument("--calibration")
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--teacher-scale", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--log-std", type=float)
    p = sub.add_parser("export")
    p.add_argument("checkpoint")
    p.add_argument("--output", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("--output", default="runs/calibration.npz")
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--closed-loop", action="store_true")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--task", choices=list(TASKS), default="visual")
    p = sub.add_parser("evaluate")
    p.add_argument("--policy", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seconds", type=float, help="default: per-task evaluation length")
    p.add_argument("--output", default="runs/evaluation.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--hover-episodes", type=int, default=5)
    p.add_argument("--hover-seconds", type=float, default=30)
    p.add_argument("--task", choices=list(TASKS), default="visual")
    p.add_argument("--encoder", help="learned encoder .pt (free roam)")
    p = sub.add_parser("roam-feasibility")
    p.add_argument("--output", default="runs/roam-feasibility.json")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--level", type=int, default=3)
    p = sub.add_parser("roam-collect")
    p.add_argument("--output", required=True)
    p.add_argument("--flights", type=int, default=128)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--student", help="actor.json flown with probability 1 - beta")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed-base", type=int, default=200)
    p.add_argument(
        "--levels", type=int, nargs="+", help="arena levels to collect (default: all)"
    )
    p.add_argument(
        "--encoder", help="learned encoder .pt driving the brain (default: v4)"
    )
    p = sub.add_parser("roam-fit")
    p.add_argument("data", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    p = sub.add_parser("roam-step-response")
    p.add_argument("--output", default="runs/roam/step-response.json")
    p = sub.add_parser("roam-screen")
    p.add_argument(
        "controllers", nargs="+", help="teacher, cue_script, random, or a path"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--ablations", nargs="+", default=["none"])
    p.add_argument("--seed-base", type=int, default=5000)
    p.add_argument(
        "--encoder", help="learned encoder .pt for policy/bypass controllers"
    )
    p = sub.add_parser("encoder-collect")
    p.add_argument("--output", required=True)
    p.add_argument("--flights", type=int, default=32)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed-base", type=int, default=600)
    p.add_argument("--level", type=int, default=3)
    p = sub.add_parser("encoder-clone")
    p.add_argument("data", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p = sub.add_parser("sac-init-decoder")
    p.add_argument("data", nargs="+", help="stage-1 DAgger .npz files")
    p.add_argument("--encoder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p = sub.add_parser("sac-round")
    p.add_argument("learner", choices=["encoder", "decoder", "bypass"])
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--decoder", help="frozen decoder actor.json (encoder learner)")
    p.add_argument("--encoder", help="frozen encoder .pt (decoder/bypass learner)")
    p.add_argument("--init", help="previous round .zip, or the clone encoder.pt")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument(
        "--n-step",
        type=int,
        default=1,
        help="n-step returns in the Bellman target (default 1; the accepted runs used 1)",
    )
    p.add_argument(
        "--optimize-memory",
        action="store_true",
        help="store the replay buffer once and derive next_obs, halving its RAM",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue a round from its resume/ snapshot instead of starting over",
    )
    p.add_argument(
        "--actor-warmup",
        type=int,
        default=None,
        help="frames of critic-only training at the start of the round "
        "(default sac.ACTOR_WARMUP_FRAMES = 50000; 0 disables)",
    )
    p = sub.add_parser("sac-export")
    p.add_argument(
        "checkpoint",
        nargs="?",
        help="SAC .zip (round output or CheckpointCallback checkpoint)",
    )
    p.add_argument("--learner", choices=["encoder", "decoder"])
    p.add_argument(
        "--encoder", help="frozen encoder .pt (decoder learner, or --repin-decoder)"
    )
    p.add_argument(
        "--repin-decoder", help="frozen decoder JSON to re-pin onto --encoder"
    )
    p.add_argument("--output", required=True)
    p = sub.add_parser("sac-validate")
    p.add_argument("--decoder", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=9000)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("current-pointer")
    p.add_argument("--task", default="free_roam", choices=["free_roam"])
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("encoder-checks")
    p.add_argument("--policy", required=True)
    p.add_argument("--encoder", help="omit for the v4 baseline")
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--controller", choices=["teacher", "policy"], default="teacher")
    args = parser.parse_args()
    if args.command in (
        "encoder-collect",
        "encoder-clone",
        "sac-init-decoder",
        "sac-round",
        "sac-export",
        "sac-validate",
        "encoder-checks",
    ):
        from . import encoder, roam_eval, sac

        if args.command == "encoder-collect":
            result = encoder.collect_clone(
                args.output,
                args.flights,
                args.seconds,
                args.workers,
                args.seed_base,
                args.level,
            )
        elif args.command == "encoder-clone":
            result = encoder.fit_clone(args.data, args.output, args.steps)
        elif args.command == "sac-init-decoder":
            result = sac.init_decoder(args.data, args.encoder, args.output, args.steps)
        elif args.command == "sac-round":
            result = sac.train_round(
                args.learner,
                args.output,
                args.frames,
                decoder=args.decoder,
                encoder=args.encoder,
                init=args.init,
                workers=args.workers,
                seed=args.seed,
                buffer_size=args.buffer_size,
                actor_warmup=(
                    sac.ACTOR_WARMUP_FRAMES
                    if args.actor_warmup is None
                    else args.actor_warmup
                ),
                n_step=args.n_step,
                optimize_memory=args.optimize_memory,
                resume=args.resume,
            )
        elif args.command == "sac-export":
            if args.repin_decoder:
                result = sac.repin_decoder(
                    args.repin_decoder, args.encoder, args.output
                )
            elif args.learner is None:
                raise SystemExit("sac-export needs --learner or --repin-decoder")
            else:
                if not args.checkpoint:
                    raise SystemExit("sac-export needs a checkpoint .zip")
                result = sac.export_checkpoint(
                    args.checkpoint, args.learner, args.output, encoder=args.encoder
                )
        elif args.command == "sac-validate":
            result = sac.validate(
                args.decoder,
                args.encoder,
                args.output,
                args.seeds,
                args.seed_base,
                args.seconds,
                args.workers,
            )
        else:
            result = roam_eval.encoder_checks(
                args.policy,
                args.output,
                args.encoder,
                args.episodes,
                args.seconds,
                args.workers,
                controller=args.controller,
            )
            result = {k: result[k] for k in ("frames", "E1", "E2")}
        print(json.dumps(result, indent=2))
        return
    if args.command in ("roam-collect", "roam-fit", "roam-screen"):
        from . import distill

        if args.command == "roam-collect":
            result = distill.collect(
                args.output,
                args.flights,
                args.seconds,
                args.student,
                args.beta,
                args.noise,
                workers=args.workers,
                seed_base=args.seed_base,
                levels=args.levels,
                encoder=args.encoder,
            )
        elif args.command == "roam-fit":
            result = distill.fit(args.data, args.output, args.net_arch, args.steps)
        else:
            controllers = [
                c
                if c in ("teacher", "cue_script", "random")
                else f"bypass:{Path(c).resolve()}"
                if c.endswith(".zip")
                else f"policy:{Path(c).resolve()}"
                for c in args.controllers
            ]
            report = distill.screen(
                controllers,
                args.output,
                args.seeds,
                args.seconds,
                args.level,
                args.workers,
                seed_base=args.seed_base,
                ablations=args.ablations,
                encoder=args.encoder,
            )
            result = {
                k: {m: v for m, v in r.items() if m != "runs"}
                for k, r in report["results"].items()
            }
        print(json.dumps(result, indent=2))
    elif args.command == "current-pointer":
        from . import current_pointer

        payload = current_pointer.update(args.task, dry_run=args.dry_run)
        print(json.dumps(payload, indent=2))
    elif args.command == "roam-feasibility":
        from .feasibility import run

        report = run(args.output, args.episodes, args.seconds, args.workers, args.level)
        print(
            json.dumps({"sensors": report["sensors"], "gate": report["gate"]}, indent=2)
        )
    elif args.command == "roam-step-response":
        from .stabiliser import run

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        report = run(args.output)
        summary = {
            axis: {k: v for k, v in report[axis].items() if k != "trace_mps"}
            for axis in ("lateral", "vertical")
        }
        print(json.dumps(summary, indent=2))
    elif args.command == "serve":
        import uvicorn

        from .server import accepted_policies, make_app

        task_policies = dict(item.split("=", 1) for item in args.task_policy)
        policy, looming_policy = args.policy, args.looming_policy
        if args.accepted:
            # Without a decoder every command is zero and the drone holds still.
            present, missing = accepted_policies()
            for task, path in missing.items():
                print(f"accepted {task} policy not found locally: {path}", flush=True)
            policy = policy or present.pop("visual", None)
            looming_policy = looming_policy or present.pop("looming", None)
            for task, path in present.items():
                task_policies.setdefault(task, path)
        if args.current:
            from . import current_pointer

            present, missing = current_pointer.current_policies()
            for task, path in missing.items():
                print(f"current {task} policy not found locally: {path}", flush=True)
            for task, path in present.items():
                entry = current_pointer.current_entry(task)
                if entry and entry.get("encoder"):
                    print(
                        f"current {task} policy needs a learned-encoder pairing "
                        "`serve` does not support yet; skipping",
                        flush=True,
                    )
                    continue
                task_policies.setdefault(task, path)
        uvicorn.run(
            make_app(policy, looming_policy, task_policies, args.port),
            host="127.0.0.1",
            port=args.port,
        )
    elif args.command == "assay":
        from .assay import sensory_assay

        report = sensory_assay(args.output)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    elif args.command == "baseline":
        import numpy as np

        from .plant import DronePlant

        plant = DronePlant(vision=False)
        zs = []
        start = time.perf_counter()
        viewer = None
        try:
            if args.viewer:
                import mujoco.viewer

                viewer = mujoco.viewer.launch_passive(plant.model, plant.data)
            for _ in range(int(args.seconds * 200)):
                before = time.perf_counter()
                plant.advance(np.zeros(4))
                zs.append(plant.pos[0, 2])
                if viewer:
                    viewer.sync()
                    time.sleep(max(0, 0.005 - (time.perf_counter() - before)))
            report = {
                "seconds": args.seconds,
                "altitude_rms": float(np.sqrt(np.mean((np.array(zs) - 1) ** 2))),
                "real_time_factor": args.seconds / (time.perf_counter() - start),
                "state": plant.state(),
                "controller": "PID baseline; no brain commands",
            }
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
        finally:
            if viewer:
                viewer.close()
            plant.close()
    elif args.command == "calibrate":
        from .calibration import collect, collect_closed_loop

        if args.closed_loop:
            collect_closed_loop(args.output, args.trials, args.frames, task=args.task)
        else:
            collect(args.output, args.trials)
    elif args.command == "train":
        from .training import train

        train(
            args.steps,
            args.output,
            args.task,
            args.resume,
            args.calibration,
            args.envs,
            args.teacher_scale,
            args.seed,
            args.learning_rate,
            args.log_std,
        )
    elif args.command == "export":
        from .training import export_checkpoint

        print(
            json.dumps(
                {"export_max_error": export_checkpoint(args.checkpoint, args.output)}
            )
        )
    elif args.task == "free_roam":
        from .roam_eval import evaluate_free_roam

        report = evaluate_free_roam(
            args.policy,
            args.output,
            args.episodes,
            args.seconds or 120,
            args.workers,
            encoder=args.encoder,
        )
        print(json.dumps(report["acceptance"], indent=2))
    else:
        from .training import evaluate

        if args.encoder:
            print(
                "warning: --encoder is ignored (only --task free_roam flies a "
                "learned encoder)",
                file=sys.stderr,
            )
        print(
            json.dumps(
                evaluate(
                    args.policy,
                    args.episodes,
                    args.output,
                    args.seconds,
                    args.workers,
                    args.hover_episodes,
                    args.hover_seconds,
                    args.task,
                )["acceptance"],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
