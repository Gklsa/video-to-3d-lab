"""Download the pinned public checkpoint with verified, resumable HTTP ranges."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import os
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "ckpt"
REVISION = "5280ca2ea4c8cf41ac174645fd3aada430af7f40"
BASE = f"https://huggingface.co/mose3-tracker/MoSE3/resolve/{REVISION}"
SIZE = 6161332064
SHA256 = "fcbd51ff01c2bc2f76ca9db3037e0a4055f6bc5ea202a5e1a9f57a91107d183c"
WORKERS = 8


def verify(path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while block := fp.read(8 * 1024 * 1024):
            digest.update(block)
    if path.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        raise RuntimeError("Checkpoint size/SHA256 mismatch; preserving file for inspection")
    return digest.hexdigest()


def main():
    CKPT.mkdir(exist_ok=True)
    final = CKPT / "model.safetensors"
    part = CKPT / "model.safetensors.part"
    ledger = CKPT / "download_ranges.json"
    if final.exists() and final.stat().st_size == SIZE and not part.exists():
        print(f"Checkpoint verified: {verify(final)}", flush=True)
        # A checkpoint copied from the official host may have no local range
        # ledger. Inference consumes this metadata even when no download was
        # needed. Only mark the file complete after the pinned SHA256 passed.
        state = {"revision": REVISION, "size": SIZE, "prefix": 0,
                 "ranges": [{"start": 0, "next": SIZE, "end": SIZE}]}
        ledger.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if not (CKPT / "config.json").is_file():
            urllib.request.urlretrieve(f"{BASE}/config.json", CKPT / "config.json")
        return
    if ledger.exists():
        state = json.loads(ledger.read_text())
    else:
        prefix = final.stat().st_size if final.exists() else 0
        if final.exists():
            os.replace(final, part)
        if not part.exists():
            part.touch()
        chunk = (SIZE - prefix + WORKERS - 1) // WORKERS
        state = {"revision": REVISION, "size": SIZE, "prefix": prefix,
                 "ranges": [{"start": s, "next": s, "end": min(s + chunk, SIZE)}
                            for s in range(prefix, SIZE, chunk)]}
        with part.open("r+b") as fp:
            fp.truncate(SIZE)
        ledger.write_text(json.dumps(state, indent=2))
    lock = threading.Lock()
    started = time.monotonic()
    initial = state["prefix"] + sum(r["next"] - r["start"] for r in state["ranges"])

    def download(item):
        for attempt in range(3):
            start, end = item["next"], item["end"]
            if start >= end:
                return
            try:
                request = urllib.request.Request(
                    f"{BASE}/model.safetensors?download=true&range_start={start}",
                    headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(request, timeout=60) as response, part.open("r+b") as fp:
                    expected = f"bytes {start}-{end-1}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server did not honour the exact byte range")
                    fp.seek(start)
                    while item["next"] < end:
                        data = response.read(min(1024 * 1024, end - item["next"]))
                        if not data:
                            raise IOError("Incomplete download range")
                        fp.write(data)
                        with lock:
                            item["next"] += len(data)
                return
            except Exception:
                if attempt == 2:
                    raise

    def progress():
        while not stopped.wait(5):
            with lock:
                count = state["prefix"] + sum(r["next"] - r["start"] for r in state["ranges"])
                ledger.write_text(json.dumps(state, indent=2))
            elapsed = max(time.monotonic() - started, 0.01)
            print(f"{count / SIZE:.1%}: {count / 1e9:.2f}/{SIZE / 1e9:.2f} GB; "
                  f"{(count-initial)/elapsed/1e6:.1f} MB/s", flush=True)

    stopped = threading.Event()
    watcher = threading.Thread(target=progress, daemon=True)
    watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for future in as_completed([pool.submit(download, r) for r in state["ranges"]]):
                future.result()
    finally:
        stopped.set()
        watcher.join()
        ledger.write_text(json.dumps(state, indent=2))
    assert all(r["next"] == r["end"] for r in state["ranges"])
    assert part.stat().st_size == SIZE
    print(f"Verifying checkpoint SHA256: {verify(part)}", flush=True)
    os.replace(part, final)
    urllib.request.urlretrieve(f"{BASE}/config.json", CKPT / "config.json")
    print("Verified complete ranges; checkpoint ready.", flush=True)


if __name__ == "__main__":
    main()
