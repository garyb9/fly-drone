import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fly connectome drone laboratory")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--policy")
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("assay")
    p.add_argument("--output", default="runs/sensory-assay.json")
    p = sub.add_parser("baseline")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--output", default="runs/baseline.json")
    p = sub.add_parser("train")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--task", choices=["hover", "visual", "looming"], default="visual")
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
    p = sub.add_parser("evaluate")
    p.add_argument("--policy", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--output", default="runs/evaluation.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--hover-episodes", type=int, default=5)
    p.add_argument("--hover-seconds", type=float, default=30)
    p.add_argument("--task", choices=["visual", "looming"], default="visual")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        from .server import make_app

        uvicorn.run(make_app(args.policy), host="127.0.0.1", port=args.port)
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
        from .calibration import collect

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
    else:
        from .training import evaluate

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
