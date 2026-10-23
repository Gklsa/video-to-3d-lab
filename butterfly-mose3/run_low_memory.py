"""Local pretrained MoSE3 inference with a documented 8 GB / torch 2.2 adapter.

Uses the official architecture and checkpoint, with reduced inference precision.
The adapter selects PyTorch's available SDPA backend, stores the backbone in
bfloat16, and keeps the small output heads in float32. No weights are trained.
"""
from pathlib import Path
import argparse
import contextlib
import enum
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deps"))
sys.path.insert(0, str(ROOT / "upstream"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def construct_meta_model(torch, model_class):
    """Avoid allocating checkpoint-sized storage while preserving tiny init constants.

    DINOv2 builds a Python drop-path schedule with torch.linspace(...).item().
    Meta has no scalar values, so that schedule alone must be computed on CPU.
    This adapter is scoped to construction and does not alter the upstream file.
    """
    from unittest.mock import patch
    original_linspace = torch.linspace

    def cpu_schedule(*values, **kwargs):
        kwargs.setdefault("device", "cpu")
        return original_linspace(*values, **kwargs)

    with torch.device("meta"), patch.object(torch, "linspace", cpu_schedule):
        return model_class()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=168)
    parser.add_argument("--refs", type=int, default=512)
    parser.add_argument("--se3-chunk", type=int, default=4096,
                        help="Query pixels fitted at once; smaller batches reduce peak memory")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/butterfly_local")
    parser.add_argument("--check", action="store_true", help="Check imports without loading weights")
    parser.add_argument("--meta-check", action="store_true",
                        help="Construct the official model on meta; no weights, forward pass or GPU allocation")
    args = parser.parse_args()
    if min(args.frames, args.width, args.height, args.refs, args.se3_chunk) < 1:
        parser.error("Frame count, size limits, reference pool and chunk size must be positive")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    # torch 2.2 uses a different public SDPA context API than the reference 2.6.
    if not hasattr(torch.nn, "attention"):
        compat = types.ModuleType("torch.nn.attention")
        class Backend(enum.Enum):
            FLASH_ATTENTION = 1
            MATH = 2
            EFFICIENT_ATTENTION = 3
        compat.SDPBackend = Backend
        compat.sdpa_kernel = lambda *_args, **_kwargs: contextlib.nullcontext()
        sys.modules["torch.nn.attention"] = compat
        torch.nn.attention = compat
    from mose3.models.mose3 import MoSE3
    from mose3.models.layers import attention
    # Use identical scaled-dot-product attention with a platform-supported backend.
    attention._sdpa = lambda q, k, v, half_dtypes: F.scaled_dot_product_attention(q, k, v)
    import inference
    # This changes only batching, retaining the authors' exact fitting method.
    inference.fit_per_pixel_se3 = functools.partial(
        inference.fit_per_pixel_se3, chunk_size=args.se3_chunk)

    if args.meta_check:
        print("Creating official MoSE3 on the meta device", flush=True)
        model = construct_meta_model(torch, MoSE3)
        entries = list(model.named_parameters()) + list(model.named_buffers())
        nonmeta = [name for name, tensor in entries if not tensor.is_meta]
        print(json.dumps({"torch": torch.__version__, "meta_construction": "ok",
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "state_entries": len(entries), "nonmeta_entries": nonmeta,
                          "cuda_allocated_bytes": torch.cuda.memory_allocated()
                          if torch.cuda.is_available() else 0}))
        if nonmeta:
            raise RuntimeError(f"Unexpected materialized entries in meta model: {nonmeta}")
        return
    if args.check:
        print(json.dumps({"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                          "numpy": np.__version__, "imports": "ok"}))
        return
    checkpoint = ROOT / "ckpt/model.safetensors"
    ranges = json.loads((ROOT / "ckpt/download_ranges.json").read_text())
    if not all(r["next"] == r["end"] for r in ranges["ranges"]):
        raise RuntimeError("Checkpoint download is incomplete; resume download_weights.py first")
    if not torch.cuda.is_available():
        raise RuntimeError("This low-memory runner requires a CUDA GPU")
    if args.output.exists():
        raise FileExistsError(f"Preserving existing run: {args.output}")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")
    started = time.monotonic()
    print("Creating model on meta device (no duplicate CPU checkpoint allocation)", flush=True)
    model = construct_meta_model(torch, MoSE3)
    fp32_prefixes = ("base.camera_head.", "base.point_head.", "base.conf_head.",
                     "track_head.", "vis_head.", "rigid_head.")
    tensors = {}
    print("Loading pretrained tensors directly to GPU; backbone bf16, output heads fp32", flush=True)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as reader:
        for i, key in enumerate(reader.keys()):
            tensor = reader.get_tensor(key)
            dtype = (torch.float32 if key.startswith(fp32_prefixes) else torch.bfloat16)
            if not tensor.is_floating_point():
                dtype = tensor.dtype
            tensors[key] = tensor.to(device=device, dtype=dtype)
            if i % 500 == 0:
                print(f"  loaded {i} tensors", flush=True)
    model.load_state_dict(tensors, strict=True, assign=True)
    del tensors
    missing = [name for name, x in list(model.named_parameters()) + list(model.named_buffers()) if x.is_meta]
    if missing:
        raise RuntimeError(f"Unmaterialized tensors: {missing}")
    model.eval()
    print(f"Loaded parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    cfg = inference.resolve_clip_config({
        "video_path": str(ROOT / "upstream/examples/butterfly.mp4"),
        "num_frames": args.frames, "frame_interval": 1,
        "max_w": args.width, "max_h": args.height,
        "num_ref_pts": args.refs, "se3_dist_weight": True,
        "output_dir": str(args.output.resolve()), "fps": 10,
    })
    payload = inference.run_clip(model, cfg, device)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    cfg["local_reproduction"] = {
        "method": "official pretrained MoSE3 inference; not retraining",
        "checkpoint_revision": ranges["revision"],
        "input_sha256": hashlib.sha256(Path(cfg["video_path"]).read_bytes()).hexdigest(),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(),
        "backbone_dtype": "bfloat16", "output_heads_dtype": "float32",
        "attention": "same SDPA operator; automatic platform backend selection",
        "inference_protocol": {
            "processed_shape": list(payload["images"].shape[:3]),
            "num_ref_pts": args.refs,
            "se3_chunk_size": args.se3_chunk,
            "storage_precision_adapted": True,
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip(),
        "ground_truth_available": False,
    }
    inference.save_clip_outputs(args.output, cfg, payload)
    print(f"SUCCESS: {args.output / 'predictions.npz'} ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
