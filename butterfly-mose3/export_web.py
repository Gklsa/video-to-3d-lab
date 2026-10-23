"""Export saved MoSE3 predictions for a small, dependency-free WebGL viewer.

This reads an existing predictions.npz. It does not invoke inference, modify
trajectories, interpolate frames, or infer unseen geometry. All binary arrays
are little-endian, tightly packed, C-order, without headers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True,
                        help="Local MoSE3 project with upstream/ and deps/; no machine-specific default.")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "web")
    parser.add_argument("--points", type=int, default=40000)
    parser.add_argument("--max-bytes", type=int, default=20_000_000,
                        help="Maximum total binary bytes; increase explicitly for denser actual sampling.")
    parser.add_argument("--axes", type=int, default=24)
    parser.add_argument("--query-roi", type=float, nargs=4, default=(0.04, 0.10, 0.87, 1.0))
    args = parser.parse_args()
    if args.points < 1 or args.axes < 1:
        parser.error("Point and axis limits must be positive.")
    left, top, right, bottom = args.query_roi
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        parser.error("The ROI must satisfy 0 <= left < right <= 1, likewise top/bottom.")
    project = args.project.resolve()
    source = (args.predictions or project / "outputs/butterfly_refined/predictions.npz").resolve()
    output = args.out.resolve()
    sys.path[:0] = [str(project / "deps"), str(project / "upstream")]
    import numpy as np
    from visualize import (depth_edges, grid_points, pca_rgb, query_mask,
                           se3_apply, surface_normal_frames)

    with np.load(source, allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    tracks = z["tracks"]
    if tracks.ndim != 4 or tracks.shape[-1] != 3:
        raise ValueError("Expected author tracks with shape [T,H,W,3].")
    T, H, W, _ = tracks.shape
    q = int(z["query_idx"])
    if not 0 <= q < T:
        raise ValueError("Invalid query_idx.")
    if z["images"].shape != (T, H, W, 3) or z["images"].dtype != np.uint8:
        raise ValueError("Expected the original uint8 input frames.")
    if z["visibility"].shape != (T, H, W):
        raise ValueError("Visibility shape does not match point identities.")

    # Use the authors' relative depth-edge mask. Apply one query-frame crop,
    # then sample fixed pixel identities once; never resample per frame.
    yy, xx = np.indices((H, W))
    crop = ((xx >= left * W) & (xx < right * W)
            & (yy >= top * H) & (yy < bottom * H))
    official_mask = query_mask(z, edge_rtol=0.05)
    pose = z["camera_poses"][q]
    query_depth = ((tracks[q] - pose[:3, 3]) @ pose[:3, :3])[..., 2]
    all_finite = np.isfinite(tracks).all(axis=(0, 3))
    mask = official_mask & crop & all_finite & np.isfinite(query_depth) & (query_depth > 0)
    candidates = np.flatnonzero(mask)
    if not len(candidates):
        raise ValueError("The fixed query mask removed every trajectory.")
    if len(candidates) > args.points:
        ids = candidates[np.linspace(0, len(candidates) - 1, args.points, dtype=np.int64)]
    else:
        ids = candidates
    N = len(ids)
    positions = np.ascontiguousarray(tracks.reshape(T, H * W, 3)[:, ids], dtype="<f4")
    rgb = np.ascontiguousarray(z["images"][q].reshape(H * W, 3)[ids], dtype="u1")
    # Official PCA: normalizes raw embedding vectors, fits on the query frame,
    # fixes component signs, and stretches the 1st--99th percentiles.
    rigidity_image = pca_rgb(z["rigidity_emb"][q:q + 1], seed=0)[0]
    rigidity = np.ascontiguousarray(rigidity_image.reshape(H * W, 3)[ids], dtype="u1")
    logits = z["visibility"].reshape(T, H * W)[:, ids]
    visible = np.ascontiguousarray(np.isfinite(logits) & (logits >= 0), dtype="u1")

    # Sparse SE(3) glyphs use the same surface-normal convention as the
    # authors' Viser viewer. These origins are R*x_query+t, not point tracks.
    axis_grid = grid_points(H, W, 48)
    axis_mask = query_mask(z, edge_rtol=0.05, edge_margin=4) & crop & all_finite
    axis_grid = axis_grid[axis_mask[axis_grid[:, 0], axis_grid[:, 1]]]
    if len(axis_grid) > args.axes:
        axis_grid = axis_grid[np.linspace(0, len(axis_grid) - 1, args.axes, dtype=np.int64)]
    ah, aw = axis_grid.T
    origins, rotations = se3_apply(z["se3_quat"][:, ah, aw], z["se3_trans"][:, ah, aw],
                                   tracks[q, ah, aw])
    starting_frames, normal_ok = surface_normal_frames(
        tracks[q], pose[:3, 3], axis_grid, radius=12, edge=depth_edges(z, 0.05))
    rotations = rotations @ starting_frames[None]
    directions = np.swapaxes(rotations, -1, -2)
    axis_logits = z["visibility"][:, ah, aw]
    axis_visible = (z["se3_valid"][:, ah, aw] & normal_ok[None]
                    & np.isfinite(origins).all(axis=-1)
                    & np.isfinite(directions).all(axis=(-1, -2))
                    & np.isfinite(axis_logits) & (axis_logits >= 0))
    if not np.isfinite(origins).all() or not np.isfinite(directions).all():
        raise ValueError("Nonfinite sparse axes require investigation before export.")

    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    files = {}

    def save(key: str, name: str, array, dtype: str, semantic: str) -> None:
        value = np.ascontiguousarray(array, dtype=dtype)
        path = output / "data" / name
        value.tofile(path)
        files[key] = {"file": f"data/{name}", "dtype": dtype,
                      "shape": list(value.shape), "byte_length": path.stat().st_size,
                      "sha256": digest(path), "semantic": semantic}

    save("tracks", "tracks.f32.bin", positions, "<f4", "[frame, point, xyz], raw world coordinates")
    save("rgb", "rgb.u8.bin", rgb, "u1", "[point, rgb], original query-frame color")
    save("rigidity", "rigidity.u8.bin", rigidity, "u1", "[point, rgb], official query-frame embedding PCA")
    save("visibility", "visibility.u8.bin", visible, "u1", "[frame, point], 1 = finite visibility logit >= 0")
    save("query_ids", "query_ids.u32.bin", ids, "<u4", "[point], query pixel index h*width+w")
    save("axes_origins", "axes_origins.f32.bin", origins, "<f4", "[frame, axis, xyz], SE(3) applied query anchors")
    save("axes_directions", "axes_directions.f32.bin", directions, "<f4", "[frame, axis, x/y/z axis, xyz vector]")
    save("axes_visible", "axes_visible.u8.bin", axis_visible, "u1", "[frame, axis], valid SE(3), normal frame and visibility")
    save("axes_query_ids", "axes_query_ids.u32.bin", ah * W + aw, "<u4", "[axis], query pixel index h*width+w")

    flat = positions.reshape(-1, 3)
    low, high = np.percentile(flat, [0.5, 99.5], axis=0)
    bbox_min, bbox_max = flat.min(axis=0), flat.max(axis=0)
    center = (low + high) * 0.5
    query_bbox_min = positions[q].min(axis=0)
    query_bbox_max = positions[q].max(axis=0)
    median_depth = float(np.median(query_depth[mask]))
    config_path = source.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    run = config.get("local_reproduction", {})
    total_bytes = sum(entry["byte_length"] for entry in files.values())
    manifest = {
        "format": "mose3-static-point-tracks-v1", "byte_order": "little", "array_order": "C",
        "view_crop": True,
        "frames": T, "points": N, "axes": len(axis_grid), "query_idx": q,
        "frame_count": T, "point_count": N, "axis_count": len(axis_grid),
        "width": W, "height": H, "axis_length": 0.018,
        "fps": float(config.get("fps", 10)), "input_resolution": {"width": W, "height": H},
        "coordinates": {"x": "right", "y": "down", "z": "forward",
                        "space": "raw predicted world = first-camera frame", "normalized": False,
                        "scale": "model-estimated; no metric ground truth asserted",
                        "quaternion_order_in_source": "wxyz"},
        "selection": {"query_roi": list(args.query_roi), "edge_rtol": 0.05,
                      "author_query_mask_points": int(official_mask.sum()),
                      "candidates_after_crop_finite_positive_depth": len(candidates),
                      "point_sampling": "deterministic evenly spaced row-major query pixel IDs; fixed over every frame",
                      "finite_over_all_frames": True,
                      "visibility": "original per-frame logit >= 0; export includes every selected trajectory and a mask",
                      "axes": "48px grid, max 24 deterministic samples; author surface_normal_frames(radius=12, edge_margin=4)"},
        "bounds": {"all_frames_min": bbox_min.tolist(), "all_frames_max": bbox_max.tolist(),
                   "robust_p005": low.tolist(), "robust_p995": high.tolist(), "robust_center": center.tolist()},
        "initial_camera": {"position": [0.0, 0.0, 0.0], "look_at": [0.0, 0.0, median_depth],
                           "up": [0.0, -1.0, 0.0],
                           "vertical_fov_radians": float(2 * np.arctan(0.5 * H / z["intrinsics"][1, 1])),
                           "source_aspect": W / H},
        "intrinsics": z["intrinsics"].tolist(), "camera_poses": z["camera_poses"].tolist(),
        "camera": {"eye": [0.0, 0.0, 0.0], "target": [0.0, 0.0, median_depth],
                   "up": [0.0, -1.0, 0.0],
                   "vertical_fov_rad": float(2 * np.arctan(0.5 * H / z["intrinsics"][1, 1]))},
        "bbox": {"min": query_bbox_min.tolist(), "max": query_bbox_max.tolist()},
        "files": {key: entry["file"] for key, entry in files.items()},
        "arrays": files, "total_binary_bytes": total_bytes,
        "validation": {"positions_finite": bool(np.isfinite(positions).all()),
                       "unique_query_ids": len(np.unique(ids)) == N,
                       "source_track_values_exact_float32": bool(np.array_equal(positions, tracks.reshape(T, H * W, 3)[:, ids])),
                       "visibility_counts_per_frame": visible.sum(axis=1).tolist(),
                       "visible_fraction_query": float(visible[q].mean()),
                       "visible_fraction_all_frames": float(visible.mean()),
                       "axes_visible_counts_per_frame": axis_visible.sum(axis=1).tolist()},
        "provenance": {"method": "export of actual saved pretrained MoSE3 inference; no retraining or interpolation",
                       "example": "authors' butterfly.mp4", "source_filename": source.name,
                       "source_sha256": digest(source), "source_bytes": source.stat().st_size,
                       "input_sha256": run.get("input_sha256"), "upstream_commit": run.get("upstream_commit"),
                       "checkpoint_revision": run.get("checkpoint_revision"),
                       "upstream": "https://github.com/mose3-tracker/MoSE3",
                       "exporter_sha256": digest(Path(__file__))},
    }
    if total_bytes > args.max_bytes:
        raise ValueError(f"Export exceeds the requested byte budget: {total_bytes} > {args.max_bytes}.")
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Read every produced file back and compare the stored values, including
    # lengths, before reporting success to a static viewer consumer.
    for key, entry in files.items():
        readback = np.fromfile(output / entry["file"], dtype=entry["dtype"])
        expected_count = int(np.prod(entry["shape"]))
        if readback.size != expected_count:
            raise ValueError(f"Readback size mismatch for {key}.")
        if digest(output / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Readback digest mismatch for {key}.")
    print(json.dumps({"manifest": str(path), "frames": T, "points": N, "axes": len(axis_grid),
                      "total_binary_bytes": total_bytes, "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
