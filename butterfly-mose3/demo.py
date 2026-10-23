"""Run, export or interactively view the local MoSE3 butterfly reproduction."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("download", "infer", "render", "view"))
    p.add_argument("--clip", type=Path, default=ROOT / "outputs/butterfly_refined")
    p.add_argument("--port", type=int, default=8870)
    args, extra = p.parse_known_args()
    clip = args.clip.resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "deps"), str(ROOT / "upstream"), env.get("PYTHONPATH", "")]
    )
    if args.mode == "download":
        command = [str(ROOT / "download_weights.py"), *extra]
    elif args.mode == "infer":
        command = [str(ROOT / "run_low_memory.py"), "--output", str(clip), *extra]
    else:
        predictions = clip / "predictions.npz"
        if not predictions.is_file():
            p.error(f"No local predictions at {predictions}. Run inference first.")
        if args.mode == "render":
            command = [str(ROOT / "tools/render_results.py"),
                       "--predictions", str(predictions), *extra]
        else:
            command = [str(ROOT / "view_local.py"),
                       "--clip", str(clip), "--port", str(args.port),
                       "--stride", "72", "--point_stride", "1",
                       "--axis_len", "0.018", "--axis_width", "2",
                       "--head_size", "0.0015", "--trail_width", "1.8", *extra]
            print(f"Local pretrained inference viewer: http://127.0.0.1:{args.port}", flush=True)
    return subprocess.call([sys.executable, "-X", "utf8", "-u", *command],
                           cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
