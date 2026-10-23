"""Open the saved butterfly reconstruction without downloading or running inference."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "butterfly-mose3"


def choose_clip(requested):
    if requested:
        path = Path(requested)
        return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()
    candidates = [PROJECT / "outputs" / name for name in
                  ("butterfly_refined", "butterfly_run", "butterfly_repeat")]
    return next((path for path in candidates if (path / "predictions.npz").is_file()), candidates[0])


def choose_port(start):
    for port in range(start, min(start + 20, 65536)):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free viewer port found. Close an older viewer and try again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", help="Output folder relative to butterfly-mose3")
    parser.add_argument("--port", type=int, default=8871)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check saved results and imports without starting a server")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    clip = choose_clip(args.clip)
    if not (clip / "predictions.npz").is_file():
        print("No saved reconstruction found. Follow butterfly-mose3/README.md to generate one.", file=sys.stderr)
        return 1
    if not (PROJECT / "upstream/visualize_viser.py").is_file():
        print("The MoSE3 source checkout is missing. Follow the setup section in butterfly-mose3/README.md.", file=sys.stderr)
        return 1
    sys.path.insert(0, str(PROJECT / "deps"))
    try:
        import numpy
        import viser
        import cv2
        import imageio.v2
    except ImportError:
        print("Viewer dependencies are missing in this Python environment. Follow butterfly-mose3/README.md.", file=sys.stderr)
        return 1
    if args.check:
        print(json.dumps({"python": sys.executable, "clip": str(clip),
                          "numpy": numpy.__version__, "viser": viser.__version__,
                          "saved_result_present": True}))
        return 0
    port = choose_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    print(f"Opening saved results: {clip.name}")
    print("Keep this window open. Close it or press Ctrl+C to stop the viewer.")
    command = [sys.executable, "-X", "utf8", "-u", str(PROJECT / "view_local.py"),
               "--clip", str(clip), "--port", str(port),
               "--stride", "72", "--point_stride", "1",
               "--axis_len", "0.018", "--axis_width", "2",
               "--head_size", "0.0015", "--trail_width", "1.8"]
    child = subprocess.Popen(command, cwd=PROJECT)
    try:
        deadline = time.monotonic() + 60
        while child.poll() is None:
            try:
                with urllib.request.urlopen(url, timeout=0.5) as response:
                    response.read()
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= deadline:
                    print(f"The viewer is still loading. Once ready, open {url}", flush=True)
                    return child.wait()
                time.sleep(0.2)
                continue
            print(f"Viewer ready: {url}", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            return child.wait()
        return child.returncode
    except KeyboardInterrupt:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
