"""Render genuine MoSE3 predictions as an annotated 3D video, GIF and PNG.

This tool does not run a model, invent geometry or reproduce training. Every
displayed 3D point and local SE(3) frame comes from the supplied predictions.npz.
Run --help without installing the rendering dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path,
                   help="An actual predictions.npz produced by MoSE3 inference.")
    p.add_argument("--out", type=Path, help="Default: <prediction folder>/rendered.")
    p.add_argument("--colors", choices=("rgb", "embedding", "both"), default="both")
    p.add_argument("--formats", default="mp4,gif,png", help="Comma-separated mp4,gif,png.")
    p.add_argument("--fps", type=float, default=10.0, help="Export playback FPS, not source FPS.")
    p.add_argument("--point-stride", type=int, default=1,
                   help="Sample real query pixels; 1 keeps the native predicted grid.")
    p.add_argument("--max-points", type=int, default=100000)
    p.add_argument("--point-size", type=float, default=0.85,
                   help="Matplotlib point area in pt^2; does not alter 3D point locations.")
    p.add_argument("--axis-stride", type=int, default=48)
    p.add_argument("--max-axes", type=int, default=16)
    p.add_argument("--axis-width", type=float, default=0.85)
    p.add_argument("--trail", type=int, default=7)
    p.add_argument("--max-trails", type=int, default=8)
    p.add_argument("--trail-width", type=float, default=1.0)
    p.add_argument("--axis-length", type=float, help="World units; default 2.8%% of scene extent.")
    p.add_argument("--depth-quantiles", type=float, nargs=2, default=(0.5, 99.5),
                   metavar=("LOW", "HIGH"), help="Query-camera positive-depth percentiles.")
    p.add_argument("--edge-rtol", type=float, default=0.0,
                   help="Optional relative adjacent depth-jump filter; 0 disables it.")
    p.add_argument("--query-roi", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                   help="Optional normalized query-frame crop. Point IDs stay unchanged.")
    p.add_argument("--mask", type=Path, help="Optional query-frame mask; nonzero pixels are retained.")
    p.add_argument("--width", type=int, default=1920, help="Export width; distinct from model input resolution.")
    p.add_argument("--height", type=int, default=1080, help="Export height; no super-resolution is applied.")
    p.add_argument("--crf", type=int, default=17, help="MP4 H.264 quality (lower is higher quality).")
    p.add_argument("--elev", type=float, default=12.0)
    p.add_argument("--azim", type=float, default=-88.0)
    p.add_argument("--zoom", type=float, default=1.72,
                   help="Fixed 3D camera zoom; does not change the predicted geometry.")
    p.add_argument("--title", default="Butterfly | 3D motion from video")
    p.add_argument("--provenance", type=Path, help="Optional local inference provenance JSON.")
    return p


def validate_args(a: argparse.Namespace) -> None:
    if not a.predictions.is_file():
        raise FileNotFoundError(f"Real prediction file not found: {a.predictions}")
    if min(a.point_stride, a.axis_stride, a.max_points, a.max_axes, a.max_trails) < 1:
        raise ValueError("Strides and maximum point/axis counts must be positive.")
    if a.fps <= 0 or a.width < 400 or a.height < 300 or a.trail < 0 or a.zoom <= 0:
        raise ValueError("Invalid playback FPS, image size or trail length.")
    if min(a.point_size, a.axis_width, a.trail_width) <= 0 or not 0 <= a.crf <= 51:
        raise ValueError("Point/line sizes must be positive; MP4 CRF must be in [0,51].")
    lo, hi = a.depth_quantiles
    if not 0 <= lo < hi <= 100:
        raise ValueError("Depth quantiles must satisfy 0 <= LOW < HIGH <= 100.")
    if a.query_roi and not (0 <= a.query_roi[0] < a.query_roi[2] <= 1 and
                            0 <= a.query_roi[1] < a.query_roi[3] <= 1):
        raise ValueError("ROI must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1.")
    formats = set(a.formats.split(","))
    if not formats or not formats <= {"mp4", "gif", "png"}:
        raise ValueError("Supported formats: mp4,gif,png.")


def load_predictions(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        required = ("images", "tracks", "visibility", "rigidity_emb", "se3_quat",
                    "se3_trans", "se3_valid", "camera_poses", "intrinsics", "query_idx")
        missing = set(required) - set(z.files)
        if missing:
            raise ValueError(f"Missing official MoSE3 prediction keys: {sorted(missing)}")
        d = {k: z[k] for k in required}
    if d["tracks"].ndim != 4 or d["tracks"].shape[-1] != 3:
        raise ValueError("tracks must have shape [T,H,W,3].")
    t, h, w, _ = d["tracks"].shape
    expected = {"images": (t, h, w, 3), "visibility": (t, h, w),
                "se3_quat": (t, h, w, 4), "se3_trans": (t, h, w, 3),
                "se3_valid": (t, h, w), "camera_poses": (t, 4, 4), "intrinsics": (3, 3)}
    for key, shape in expected.items():
        if d[key].shape != shape:
            raise ValueError(f"{key}: expected {shape}, received {d[key].shape}.")
    if d["rigidity_emb"].shape[:3] != (t, h, w) or d["rigidity_emb"].ndim != 4:
        raise ValueError("rigidity_emb must have shape [T,H,W,D].")
    q = int(np.asarray(d["query_idx"]).item())
    if not 0 <= q < t or min(t, h, w) < 1:
        raise ValueError("Invalid query index or empty prediction arrays.")
    if d["images"].dtype != np.uint8:
        raise ValueError("Expected original uint8 images in predictions.npz.")
    d["query_idx"] = q
    return d


def quat_to_rot(q: Any) -> tuple[Any, Any]:
    """Official inference saves wxyz quaternions; use their matrix columns as axes."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    valid = np.isfinite(q).all(axis=-1) & (n[..., 0] > 1e-10)
    q = np.divide(q, n, out=np.zeros_like(q), where=n > 1e-10)
    w, x, y, z = np.moveaxis(q, -1, 0)
    r = np.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                  2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                  2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)), axis=-1)
    return r.reshape(q.shape[:-1] + (3, 3)), valid


def display_coordinates(p: Any) -> Any:
    """World-camera convention x right/y down/z forward -> right/forward/up."""
    return np.asarray(p)[..., [0, 2, 1]] * np.array([1.0, 1.0, -1.0])


def query_depth(d: dict) -> Any:
    q = d["query_idx"]
    c2w = d["camera_poses"][q]
    if not np.isfinite(c2w).all():
        raise ValueError("The query camera pose is nonfinite.")
    # points are row vectors: inverse of X_world = R X_camera + t.
    return ((d["tracks"][q] - c2w[:3, 3]) @ c2w[:3, :3])[..., 2]


def stable_query_mask(d: dict, a: argparse.Namespace) -> tuple[Any, dict]:
    q = d["query_idx"]
    depth = query_depth(d)
    mask = np.isfinite(d["tracks"][q]).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    if not mask.any():
        raise ValueError("No finite query points with positive query-camera depth.")
    low, high = np.percentile(depth[mask], a.depth_quantiles)
    mask &= (depth >= low) & (depth <= high)
    if a.edge_rtol > 0:
        # Mark both endpoints of relative depth discontinuities. No SciPy required.
        edge = np.zeros(depth.shape, dtype=bool)
        for axis in (0, 1):
            before = [slice(None)] * 2
            after = [slice(None)] * 2
            before[axis], after[axis] = slice(None, -1), slice(1, None)
            p, n = depth[tuple(before)], depth[tuple(after)]
            jump = (~np.isfinite(p) | ~np.isfinite(n) |
                    (np.abs(p - n) > a.edge_rtol * np.maximum(np.minimum(np.abs(p), np.abs(n)), 1e-8)))
            edge[tuple(before)] |= jump
            edge[tuple(after)] |= jump
        mask &= ~edge
    h, w = depth.shape
    if a.query_roi:
        left, top, right, bottom = a.query_roi
        yy, xx = np.indices((h, w))
        mask &= (xx >= left*w) & (xx < right*w) & (yy >= top*h) & (yy < bottom*h)
    if a.mask:
        from PIL import Image
        image_mask = Image.open(a.mask).convert("L")
        image_mask = image_mask.resize((w, h), Image.Resampling.NEAREST)
        mask &= np.asarray(image_mask) > 0
    if not mask.any():
        raise ValueError("Query filtering removed all points; relax the crop, mask or depth filter.")
    return mask, {"query_mask_count": int(mask.sum()), "query_camera_depth_low": float(low),
                  "query_camera_depth_high": float(high)}


def stable_indices(mask: Any, stride: int, maximum: int) -> Any:
    grid = np.zeros_like(mask)
    grid[stride//2::stride, stride//2::stride] = True
    ids = np.flatnonzero(mask & grid)
    if not len(ids):
        ids = np.flatnonzero(mask)
    if len(ids) > maximum:
        ids = ids[np.linspace(0, len(ids)-1, maximum, dtype=int)]
    return ids


def embedding_colors(embedding: Any) -> Any:
    """A single PCA fitted on query pixels; colors stay fixed across all frames."""
    h, w, c = embedding.shape
    flat = embedding.reshape(-1, c).astype(np.float64)
    good = np.isfinite(flat).all(axis=-1)
    colors = np.full((h*w, 3), 0.18, dtype=float)
    if good.sum() < 3:
        return colors.reshape(h, w, 3)
    fit = flat[good]
    fit = fit[::max(1, len(fit)//12000)]
    center = fit.mean(axis=0)
    _, _, vt = np.linalg.svd(fit-center, full_matrices=False)
    components = vt[:3].T
    # Deterministic sign avoids arbitrary color inversion between NumPy backends.
    for k in range(components.shape[1]):
        i = np.argmax(np.abs(components[:, k]))
        if components[i, k] < 0:
            components[:, k] *= -1
    projected = (flat[good]-center) @ components
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3-projected.shape[1])))
    lo, hi = np.percentile(projected, (1, 99), axis=0)
    colors[good] = np.clip((projected-lo)/np.maximum(hi-lo, 1e-8), 0, 1)
    return colors.reshape(h, w, 3)


def summary_stats(values: Any) -> dict:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def build_scene(d: dict, a: argparse.Namespace) -> dict:
    t, h, w, _ = d["tracks"].shape
    q = d["query_idx"]
    mask, filter_stats = stable_query_mask(d, a)
    point_ids = stable_indices(mask, a.point_stride, a.max_points)
    axis_ids = stable_indices(mask, a.axis_stride, a.max_axes)
    flat = d["tracks"].reshape(t, h*w, 3)
    points = flat[:, point_ids]
    anchors = flat[q, axis_ids]
    quats = d["se3_quat"].reshape(t, h*w, 4)[:, axis_ids]
    rot, quat_valid = quat_to_rot(quats)
    trans = d["se3_trans"].reshape(t, h*w, 3)[:, axis_ids]
    axis_origins = np.einsum("tnij,nj->tni", rot, anchors) + trans
    axis_valid = (d["se3_valid"].reshape(t, h*w)[:, axis_ids].astype(bool) & quat_valid &
                  np.isfinite(axis_origins).all(axis=-1))
    axis_visible = d["visibility"].reshape(t, h*w)[:, axis_ids] > 0
    point_visible = d["visibility"].reshape(t, h*w)[:, point_ids] > 0
    tracked = flat[:, axis_ids]
    residual = np.linalg.norm(axis_origins-tracked, axis=-1)
    residual_valid = axis_valid & np.isfinite(tracked).all(axis=-1)
    residual_valid[q] = False  # Exclude the identity transform by design.
    finite = points[np.isfinite(points).all(axis=-1)]
    if not len(finite):
        raise ValueError("No finite sampled tracks to display.")
    plot_points = display_coordinates(finite)
    low, high = np.percentile(plot_points, (0.5, 99.5), axis=0)
    center = (low+high)/2
    extent = max(float((high-low).max()), 1e-4)
    half = extent*0.57
    pca = embedding_colors(d["rigidity_emb"][q])
    track_colors = plt.get_cmap("turbo")((axis_ids % w)/max(w-1, 1))[:, :3]
    finite_per_frame = [int(np.isfinite(frame).all(axis=-1).sum()) for frame in flat]
    metrics = {
        "measurement_type": "Prediction sanity checks; no ground truth and no accuracy estimate",
        "world_units": "Model-predicted coordinate units; metric scale is not validated",
        "frames": t, "height": h, "width": w, "query_idx": q,
        "finite_tracks_per_frame": finite_per_frame,
        "finite_track_fraction_per_frame": [n/(h*w) for n in finite_per_frame],
        **filter_stats, "rendered_stable_point_count": int(len(point_ids)),
        "sampled_axis_count": int(len(axis_ids)),
        "valid_axes_per_frame": axis_valid.sum(axis=1).astype(int).tolist(),
        "visible_sampled_points_per_frame": point_visible.sum(axis=1).astype(int).tolist(),
        "se3_track_consistency_residual_nonquery": summary_stats(residual[residual_valid]),
        "residual_definition": "||R(q_t) X_query + translation_t - predicted_track_t||; sampled axes, valid nonquery frames only",
        "quaternion_norm_absolute_deviation": summary_stats(np.abs(np.linalg.norm(quats, axis=-1)-1)),
        "ground_truth_available": False,
        "display_protocol": {
            "model_resolution": [w, h], "export_resolution": [a.width, a.height],
            "point_stride": a.point_stride, "point_size_pt2": a.point_size,
            "trajectory_count": min(a.max_trails, len(axis_ids)),
            "postprocess_geometry_smoothing": False,
            "interpolated_3d_points": False,
            "temporal_interpolation": False,
            "description": "Denser display of native predictions and HD composition; export resolution is not model resolution",
        },
    }
    return dict(points=display_coordinates(points), rgb=d["images"][q].reshape(-1, 3)[point_ids]/255.0,
                embedding=pca.reshape(-1, 3)[point_ids], pca=pca,
                axes_origin=display_coordinates(axis_origins),
                axes_dirs=display_coordinates(np.swapaxes(rot, -1, -2)),
                axes_valid=axis_valid, axes_visible=axis_visible, visible=point_visible,
                tracks=display_coordinates(tracked), track_colors=track_colors,
                trail_indices=np.linspace(0, len(axis_ids)-1, min(a.max_trails, len(axis_ids)), dtype=int),
                center=center, half=half, extent=extent, metrics=metrics,
                point_ids=point_ids, axis_ids=axis_ids)


def render_frame(d: dict, scene: dict, a: argparse.Namespace, t: int, mode: str) -> Any:
    background, foreground, muted = "#07111a", "#f1f5f9", "#99adbf"
    fig = plt.figure(figsize=(a.width/100, a.height/100), dpi=100, facecolor=background)
    fig.text(.028, .945, a.title, fontsize=25, color=foreground, weight="bold")
    h, w = d["images"].shape[1:3]
    fig.text(.028, .907, f"MoSE3 pretrained inference   |   Model input {w} x {h}   |   "
             f"Export {a.width} x {a.height}", fontsize=11, color=muted)
    image_ax = fig.add_axes((.028, .555, .22, .27), facecolor=background)
    image_ax.imshow(d["images"][t], interpolation="nearest")
    image_ax.set_axis_off()
    image_ax.set_title("SOURCE VIDEO", loc="left", color=foreground, fontsize=10, pad=10)
    emb_ax = fig.add_axes((.028, .215, .22, .27), facecolor=background)
    emb_ax.imshow(scene["pca"], interpolation="nearest")
    emb_ax.set_axis_off()
    emb_ax.set_title("QUERY RIGIDITY | PCA", loc="left", color=foreground, fontsize=10, pad=10)
    fig.text(.028, .169, "Local axes   X / red    Y / green    Z / blue", fontsize=9, color=muted)
    fig.text(.028, .140, "Point locations come directly from model predictions.", fontsize=9, color=muted)
    ax = fig.add_axes((.255, .075, .74, .805), projection="3d", facecolor=background)
    ax.set_proj_type("persp")
    colors = scene[mode]
    pts = scene["points"][t]
    valid = np.isfinite(pts).all(axis=-1)
    visible = valid & scene["visible"][t]
    # Occluded predictions remain explicitly faint; only finite positions are shown.
    if (valid & ~visible).any():
        p = pts[valid & ~visible]
        ax.scatter(*p.T, c=colors[valid & ~visible], s=a.point_size*.7, alpha=.075,
                   depthshade=False, linewidths=0, rasterized=True)
    if visible.any():
        p = pts[visible]
        ax.scatter(*p.T, c=colors[visible], s=a.point_size, alpha=1.0,
                   depthshade=False, linewidths=0, rasterized=True)
    origins = scene["axes_origin"][t]
    dirs = scene["axes_dirs"][t]
    axes_ok = scene["axes_valid"][t] & scene["axes_visible"][t]
    length = a.axis_length if a.axis_length is not None else scene["extent"]*.028
    endpoints = origins[:, None, :] + length*dirs
    segments = np.stack((np.broadcast_to(origins[:, None, :], endpoints.shape), endpoints), axis=-2)
    axis_colors = np.array(["#fa685f", "#70d9a3", "#66b8ff"])
    if axes_ok.any():
        ax.add_collection3d(Line3DCollection(segments[axes_ok].reshape(-1, 2, 3),
                                            colors=np.tile(axis_colors, int(axes_ok.sum())),
                                            linewidths=a.axis_width, alpha=.92))
    if a.trail > 0 and t > 0:
        start = max(0, t-a.trail)
        tr = scene["tracks"][start:t+1, scene["trail_indices"]]
        trail_segments = np.stack((tr[:-1], tr[1:]), axis=2).transpose(1, 0, 2, 3).reshape(-1, 2, 3)
        valid_trail = np.isfinite(trail_segments).all(axis=(1, 2))
        trail_colors = np.repeat(scene["track_colors"][scene["trail_indices"]], t-start, axis=0)
        ax.add_collection3d(Line3DCollection(trail_segments[valid_trail], colors=trail_colors[valid_trail],
                                            linewidths=a.trail_width, alpha=.84))
    heads = scene["tracks"][t, scene["trail_indices"]]
    good_heads = np.isfinite(heads).all(axis=-1) & scene["axes_visible"][t, scene["trail_indices"]]
    if good_heads.any():
        ax.scatter(*heads[good_heads].T, c=scene["track_colors"][scene["trail_indices"]][good_heads],
                   s=5.5, linewidths=0, depthshade=False)
    c, radius = scene["center"], scene["half"]
    ax.set(xlim=(c[0]-radius, c[0]+radius), ylim=(c[1]-radius, c[1]+radius),
           zlim=(c[2]-radius, c[2]+radius))
    ax.set_box_aspect((1, 1, 1), zoom=a.zoom)
    ax.view_init(elev=a.elev, azim=a.azim)
    ax.set_axis_off()
    ax.text2D(.10, .985, f"3D MOTION  /  {mode.upper()}", transform=ax.transAxes,
              fontsize=11, color=foreground)
    fig.text(.37, .078, f"Frame {t+1:02d}/{len(d['images']):02d}    |    "
             f"{int(valid.sum()):,} predicted points    |    {int(axes_ok.sum())} local axes",
             fontsize=10, color=foreground)
    fig.text(.028, .028, "MoSE3 authors' pretrained model + butterfly example  |  Local inference and visualization",
             fontsize=9, color=muted)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def write_outputs(d: dict, scene: dict, a: argparse.Namespace) -> dict:
    import imageio.v2 as imageio
    from PIL import Image
    out = a.out or a.predictions.parent / "rendered"
    out.mkdir(parents=True, exist_ok=True)
    requested = set(a.formats.split(","))
    modes = ("rgb", "embedding") if a.colors == "both" else (a.colors,)
    produced: list[str] = []
    failures: list[dict] = []
    for mode in modes:
        writers = {}
        for kind in ("mp4", "gif"):
            if kind not in requested:
                continue
            target = out / f"motion_{mode}.{kind}"
            try:
                if kind == "mp4":
                    writers[kind] = imageio.get_writer(target, fps=a.fps, codec="libx264", quality=None,
                                                       macro_block_size=2, pixelformat="yuv420p",
                                                       ffmpeg_params=["-crf", str(a.crf), "-preset", "medium"])
                else:
                    writers[kind] = imageio.get_writer(target, mode="I", duration=1000/a.fps, loop=0)
            except Exception as exc:
                failures.append({"format": kind, "mode": mode, "error": str(exc)})
        still_indices = {0, len(d["images"])//2, len(d["images"])-1}
        for t in range(len(d["images"])):
            frame = render_frame(d, scene, a, t, mode)
            if "png" in requested and t in still_indices:
                target = out / f"{mode}_frame_{t:03d}.png"
                Image.fromarray(frame).save(target)
                produced.append(target.name)
                if t == len(d["images"])//2:
                    cover = out / f"cover_{mode}.png"
                    Image.fromarray(frame).save(cover)
                    produced.append(cover.name)
            for kind, writer in list(writers.items()):
                try:
                    writer.append_data(frame)
                except Exception as exc:
                    failures.append({"format": kind, "mode": mode, "error": str(exc)})
                    try:
                        writer.close()
                    except Exception:
                        pass
                    del writers[kind]
                    # Never leave a partial video looking like a completed result.
                    (out / f"motion_{mode}.{kind}").unlink(missing_ok=True)
            print(f"[render] {mode}: {t+1}/{len(d['images'])}", flush=True)
        for kind, writer in writers.items():
            try:
                writer.close()
                produced.append(f"motion_{mode}.{kind}")
            except Exception as exc:
                failures.append({"format": kind, "mode": mode, "error": str(exc)})
                (out / f"motion_{mode}.{kind}").unlink(missing_ok=True)
    np.savez_compressed(out / "rendered_query_pixel_ids.npz", point_ids=scene["point_ids"],
                        axis_ids=scene["axis_ids"], height=d["images"].shape[1],
                        width=d["images"].shape[2], query_idx=d["query_idx"])
    metrics = scene["metrics"]
    digest = hashlib.sha256()
    with a.predictions.open("rb") as prediction_stream:
        for chunk in iter(lambda: prediction_stream.read(1024*1024), b""):
            digest.update(chunk)
    metrics.update({"prediction_file": str(a.predictions.resolve()),
                    "prediction_sha256": digest.hexdigest(),
                    "render_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    "produced_files": produced, "export_failures": failures,
                    "playback_fps": a.fps})
    config_path = a.predictions.parent / "config.json"
    if config_path.is_file():
        metrics["inference_config"] = json.loads(config_path.read_text(encoding="utf-8"))
    if a.provenance:
        metrics["inference_provenance"] = json.loads(a.provenance.read_text(encoding="utf-8"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    if not produced:
        raise RuntimeError(f"No requested exports were completed. See {out / 'metrics.json'}")
    return {"output_dir": str(out.resolve()), "produced": produced, "failures": failures}


def main() -> None:
    a = parser().parse_args()
    validate_args(a)
    global np, plt, Line3DCollection
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    d = load_predictions(a.predictions)
    scene = build_scene(d, a)
    print(json.dumps(write_outputs(d, scene, a), indent=2))


if __name__ == "__main__":
    main()
