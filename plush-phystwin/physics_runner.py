"""Run the official PhysTwin spring-mass simulator with released trained assets.

This adapter does not replace the physical solver or fabricate a deformation.
It avoids importing upstream's optional native Gaussian renderer.  Gaussian
motion uses the upstream Kabsch/LBS interpolation with a chunked KNN initializer.
Author data and pretrained physical parameters remain distinguishable from a
new local rollout.  Upstream sources and licenses must accompany this adapter.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import pickle
import sys
import time
import types
from pathlib import Path


CASE = "double_stretch_sloth"
GS_EXPERIMENT = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _function_from_source(path, name, namespace):
    """Execute one unchanged function from the upstream AST, not its imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one upstream {name}, found {len(matches)}")
    unit = ast.Module(body=[matches[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace[name]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PhysTwinRuntime:
    """Official trained single-object twin plus a lightweight viewer interface.

    ``step`` receives two handle translations in the source world coordinate
    frame, in metres, relative to their original locations.  Those translations
    drive kinematic *controllers*, not the object particles.  Object motion is
    integrated by the original Warp kernels and learned spring parameters.
    Ground is z=0; for this dataset gravity points toward positive world z.
    """

    def __init__(self, project_root, upstream_root=None, case_name=CASE,
                 max_gaussians=60000, enable_gaussians=True):
        self.root = Path(project_root).resolve()
        self.upstream = Path(upstream_root).resolve() if upstream_root else self.root / "upstream"
        if not self.upstream.exists():
            # The staging tree can be tested before copying into its final folder.
            candidate = Path(__file__).resolve().parents[1] / "upstream"
            if candidate.exists():
                self.upstream = candidate
        for deps in (self.root / "deps", Path(__file__).resolve().parent / "deps"):
            if deps.exists():
                sys.path.insert(0, str(deps))

        import numpy as np
        import torch
        import warp as wp
        import open3d as o3d
        import yaml
        from sklearn.cluster import KMeans

        self.np, self.torch, self.wp = np, torch, wp
        self.case_name = case_name
        self.device = "cuda:0"
        self.up_direction = "-z"
        if not torch.cuda.is_available():
            raise RuntimeError("This official PhysTwin adapter requires an NVIDIA CUDA GPU.")
        self.started = time.perf_counter()
        self.step_count = 0
        self.max_gaussians = max_gaussians
        self.solver_path = self.upstream / "qqtt/model/diff_simulator/spring_mass_warp.py"
        self.trainer_path = self.upstream / "qqtt/engine/trainer_warp.py"

        # Load the original Config without importing qqtt's optional renderer.
        cfg_file = self.upstream / "qqtt/utils/config.py"
        cfg_tree = ast.parse(cfg_file.read_text(encoding="utf-8"))
        cfg_class = next(n for n in cfg_tree.body if isinstance(n, ast.ClassDef) and n.name == "Config")
        cfg_class.decorator_list = []
        cfg_ns = {"yaml": yaml}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cfg_class], type_ignores=[])), str(cfg_file), "exec"), cfg_ns)
        cfg = cfg_ns["Config"]()
        cfg.load_from_yaml(str(self.upstream / "configs/real.yaml"))
        cfg.device = self.device
        cfg.collision_learn = False
        cfg.use_graph = True

        self.data_path = self.root / f"data/different_types/{case_name}/final_data.pkl"
        optimal_path = self.root / f"experiments_optimization/{case_name}/optimal_params.pkl"
        models = sorted((self.root / f"experiments/{case_name}/train").glob("best_*.pth"))
        if len(models) != 1:
            raise RuntimeError(f"Expected one author checkpoint in experiments/{case_name}/train, found {len(models)}")
        self.model_path = models[0]
        with optimal_path.open("rb") as handle:
            optimal = pickle.load(handle)
        cfg.set_optimal_params(dict(optimal))
        # Only load these original public author assets; do not use arbitrary pickles.
        with self.data_path.open("rb") as handle:
            data = pickle.load(handle)
        self.data = data
        self.cfg = cfg

        logger = logging.getLogger("PhysTwin")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        utils_stub = types.ModuleType("qqtt.utils")
        utils_stub.cfg, utils_stub.logger = cfg, logger
        sys.modules["qqtt.utils"] = utils_stub
        # Load original kernels from disk; Warp retains the original source file.
        solver = _load_module("phystwin_official_spring_mass", self.solver_path)
        self.solver = solver
        if not wp.is_cuda_available():
            raise RuntimeError("Warp cannot access the CUDA device. Check its wheel/driver compatibility.")

        object_points = np.asarray(data["object_points"], dtype=np.float32)
        structure = np.concatenate([object_points[0], data["surface_points"], data["interior_points"]], axis=0).astype(np.float32)
        self.initial_particles = structure.copy()
        self.num_original = object_points.shape[1]
        self.num_surface = self.num_original + len(data["surface_points"])
        self.num_particles = len(structure)
        self.initial_controls = np.asarray(data["controller_points"][0], dtype=np.float32)
        self.recorded_controls = np.asarray(data["controller_points"], dtype=np.float32)
        self.control_groups = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(self.initial_controls)
        group_centers = np.stack([self.initial_controls[self.control_groups == i].mean(0) for i in range(2)])
        if group_centers[0, 1] > group_centers[1, 1]:
            self.control_groups = 1 - self.control_groups
        self.handle_centers = np.stack([self.initial_controls[self.control_groups == i].mean(0) for i in range(2)])
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.device)
        self._tensor = tensor
        self._original_colors = np.asarray(data["object_colors"][0], dtype=np.float32)
        if self._original_colors.max() > 1.0:
            self._original_colors /= 255.0

        # Exact author Open3D neighbor search / spring ordering is important:
        # learned per-spring stiffnesses must match these edges one-for-one.
        init_start = _function_from_source(self.trainer_path, "_init_start", {"o3d": o3d, "torch": torch, "np": np, "cfg": cfg})
        vertices, springs, rests, masses, n_object_springs = init_start(
            None, tensor(structure), tensor(self.initial_controls),
            object_radius=cfg.object_radius, object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius, controller_max_neighbours=cfg.controller_max_neighbours,
        )
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if len(checkpoint["spring_Y"]) != len(springs):
            raise RuntimeError(f"Spring topology mismatch: reconstructed {len(springs)} vs trained {len(checkpoint['spring_Y'])}")
        if int(checkpoint["num_object_springs"]) != int(n_object_springs):
            raise RuntimeError("Object/controller spring boundary disagrees with author checkpoint.")
        self.springs = springs.detach().cpu().numpy()
        self.rest_lengths = rests.detach().cpu().numpy()
        self.spring_stiffness = checkpoint["spring_Y"].detach().cpu().numpy()
        self._trained_log_stiffness = torch.log(checkpoint["spring_Y"]).detach().clone()
        self._controller_spring_groups = np.full(len(self.springs), -1, dtype=np.int32)
        for endpoint in (0, 1):
            mask = self.springs[:, endpoint] >= self.num_particles
            self._controller_spring_groups[mask] = self.control_groups[self.springs[mask, endpoint] - self.num_particles]
        self.sim = solver.SpringMassSystemWarp(
            vertices, springs, rests, masses,
            dt=cfg.dt, num_substeps=cfg.num_substeps, spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas, collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping, drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas, collide_object_fric=cfg.collide_object_fric,
            collision_dist=cfg.collision_dist, num_object_points=len(structure),
            num_surface_points=self.num_surface, num_original_points=self.num_original,
            controller_points=tensor(self.recorded_controls), reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min, spring_Y_max=cfg.spring_Y_max,
            gt_object_points=tensor(object_points),
            gt_object_visibilities=tensor(data["object_visibilities"], torch.bool),
            gt_object_motions_valid=tensor(data["object_motions_valid"], torch.bool),
            self_collision=cfg.self_collision, disable_backward=True,
        )
        self.sim.set_spring_Y(torch.log(checkpoint["spring_Y"]).detach().clone())
        self.sim.set_collide(checkpoint["collide_elas"].detach().clone(), checkpoint["collide_fric"].detach().clone())
        self.sim.set_collide_object(checkpoint["collide_object_elas"].detach().clone(), checkpoint["collide_object_fric"].detach().clone())
        self.frame_dt = float(cfg.dt * cfg.num_substeps)
        self.gaussian_path = self.root / f"gaussian_output/{case_name}/{GS_EXPERIMENT}/point_cloud/iteration_10000/point_cloud.ply"
        self.gaussian_enabled = bool(enable_gaussians)
        if self.gaussian_enabled:
            self._load_gaussians(max_gaussians)
        self.reset()
        print(json.dumps({"status": "ready", "particles": self.num_particles, "springs": len(self.springs),
                          "gaussians": len(self.gaussian_colors) if self.gaussian_enabled else 0,
                          "frame_dt": self.frame_dt, "gpu": torch.cuda.get_device_name(0)}, ensure_ascii=False), flush=True)

    def _load_gaussians(self, max_gaussians):
        import numpy as np
        from plyfile import PlyData
        from scipy.spatial import cKDTree
        torch = self.torch
        ply = PlyData.read(self.gaussian_path)["vertex"]
        xyz = np.stack([ply[a] for a in ("x", "y", "z")], axis=1).astype(np.float32)
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(ply["opacity"], dtype=np.float32)))
        keep = np.flatnonzero(opacity >= 0.1)
        if max_gaussians and len(keep) > max_gaussians:
            # Deterministic coverage sample from author appearance; not new geometry.
            keep = keep[np.linspace(0, len(keep) - 1, max_gaussians, dtype=np.int64)]
        self.gaussian_indices = keep
        self.gaussian_colors = np.clip(np.stack([ply[f"f_dc_{i}"] for i in range(3)], axis=1)[keep] * 0.28209479177387814 + 0.5, 0, 1).astype(np.float32)
        self.gaussian_opacities = opacity[keep, None]
        # Official playground sets isotropic=True, using its first scale in all axes.
        scales = np.exp(np.asarray(ply["scale_0"], dtype=np.float32)[keep])
        self.gaussian_covariances = (np.eye(3, dtype=np.float32)[None] * scales[:, None, None] ** 2)
        self._initial_gaussians = self._tensor(xyz[keep])
        self._initial_rotations = self._tensor(np.stack([ply[f"rot_{i}"] for i in range(4)], axis=1)[keep])
        self._initial_rotations = torch.nn.functional.normalize(self._initial_rotations, dim=-1)
        # Same exact Euclidean KNN / inverse-distance weighting as upstream, but
        # cKDTree avoids allocating its >G*N*3 CPU tensor for initialization.
        k = min(16, self.num_particles - 1)
        tree = cKDTree(self.initial_particles)
        _, gaussian_knn = tree.query(xyz[keep], k=k, workers=-1)
        _, bone_knn = tree.query(self.initial_particles, k=k + 1, workers=-1)
        self._weights_indices = self._tensor(gaussian_knn, torch.long)
        self._relations = self._tensor(bone_knn[:, 1:], torch.long)
        self.gaussian_knn_indices = gaussian_knn.astype(np.int32)
        self._dynamic = _load_module("phystwin_official_dynamic_utils", self.upstream / "gaussian_splatting/dynamic_utils.py")

    def reset(self):
        self.sim.set_spring_Y(self._trained_log_stiffness.detach().clone())
        self.sim.set_init_state(self.sim.wp_init_vertices, self.sim.wp_init_velocities)
        self.current_controls = self._tensor(self.initial_controls.copy())
        self.current_particles = self._tensor(self.initial_particles.copy())
        self.last_handle_offsets = self.np.zeros((2, 3), dtype=self.np.float32)
        self.step_count = 0
        self.active_handles = self.np.ones(2, dtype=bool)
        if self.gaussian_enabled:
            self.current_gaussians = self._initial_gaussians.clone()
            self.current_rotations = self._initial_rotations.clone()
        self.wp.synchronize()

    def set_active_handles(self, active_handles):
        """Attach/detach controller springs through the official kernel gate.

        logK=-inf means exp(logK)==0; upstream eval_springs skips that spring
        entirely (including its dashpot).  Internal springs stay untouched.
        Re-grasping uses the original rest lengths, so reset before re-grasping
        a far-away object to avoid a large discontinuous force.
        """
        active = self.np.asarray(active_handles, dtype=bool)
        if active.shape != (2,):
            raise ValueError("active_handles must be two boolean values")
        if self.np.array_equal(active, self.active_handles):
            return
        stiffness = self._trained_log_stiffness.detach().clone()
        disabled = self.np.zeros(len(self.springs), dtype=bool)
        for group in range(2):
            if not active[group]:
                disabled |= self._controller_spring_groups == group
        stiffness[self._tensor(disabled, self.torch.bool)] = -float("inf")
        self.sim.set_spring_Y(stiffness)
        self.active_handles = active.copy()

    def step(self, handle_offsets_m=None, controller_positions=None, active_handles=None):
        torch, wp, np = self.torch, self.wp, self.np
        if active_handles is not None:
            self.set_active_handles(active_handles)
        if controller_positions is None:
            offsets = np.asarray(handle_offsets_m, dtype=np.float32)
            if offsets.shape != (2, 3) or not np.isfinite(offsets).all():
                raise ValueError("handle_offsets_m must be finite shape (2,3) in metres")
            if np.max(np.abs(offsets)) > 0.5:
                raise ValueError("Handle translations above 0.5m are outside this adapter's demonstration range")
            control = self.initial_controls + offsets[self.control_groups]
            self.last_handle_offsets = offsets.copy()
        else:
            control = np.asarray(controller_positions, dtype=np.float32)
            if control.shape != self.initial_controls.shape or not np.isfinite(control).all():
                raise ValueError("controller_positions must match the author control points")
        next_controls = self._tensor(control)
        self.sim.set_controller_interactive(self.current_controls, next_controls)
        if self.sim.object_collision_flag:
            self.sim.update_collision_graph()
        wp.capture_launch(self.sim.forward_graph)
        x = wp.to_torch(self.sim.wp_states[-1].wp_x, requires_grad=False).clone()
        if not torch.isfinite(x).all().item():
            raise FloatingPointError("Official physical solver produced non-finite positions; reduce handle step size")
        if self.gaussian_enabled:
            with torch.no_grad():
                weights = self._dynamic.calc_weights_vals_from_indices(self.current_particles, self.current_gaussians, self._weights_indices)
                self.current_gaussians, self.current_rotations, _ = self._dynamic.interpolate_motions_speedup(
                    bones=self.current_particles, motions=x - self.current_particles, relations=self._relations,
                    xyz=self.current_gaussians, quat=self.current_rotations, weights=weights,
                    weights_indices=self._weights_indices, device=self.device, step=str(self.step_count),
                )
        self.current_particles = x
        self.current_controls = next_controls
        self.sim.set_init_state(self.sim.wp_states[-1].wp_x, self.sim.wp_states[-1].wp_v)
        self.step_count += 1
        wp.synchronize()
        return self.particles()

    def particles(self):
        return self.current_particles.detach().cpu().numpy()

    def controls(self):
        return self.current_controls.detach().cpu().numpy()

    def gaussians(self):
        if not self.gaussian_enabled:
            return None
        return {"centers": self.current_gaussians.detach().cpu().numpy(),
                "covariances": self.gaussian_covariances,
                "rgbs": self.gaussian_colors, "opacities": self.gaussian_opacities}

    @property
    def simulation_time(self):
        return self.step_count * self.frame_dt

    def provenance(self):
        return {"framework": "PhysTwin", "case": self.case_name, "mode": "local forward physics with author pretrained parameters",
                "solver": "upstream qqtt/model/diff_simulator/spring_mass_warp.py (unchanged)",
                "solver_sha256": _sha256(self.solver_path), "checkpoint_sha256": _sha256(self.model_path),
                "input_sha256": _sha256(self.data_path), "dt": float(self.cfg.dt),
                "substeps": int(self.cfg.num_substeps), "frame_dt": self.frame_dt,
                "source_fps": int(self.cfg.FPS), "self_collision": bool(self.cfg.self_collision),
                "particles": self.num_particles, "springs": len(self.springs),
                "gaussian_count": len(self.gaussian_colors) if self.gaussian_enabled else 0,
                "appearance": "author 3D Gaussian assets, opacity filter >=0.1, optional deterministic subsample, DC color, isotropic covariances",
                "skinning": "official incremental Kabsch/LBS, exact Euclidean cKDTree KNN initialization adapter",
                "torch": self.torch.__version__, "warp": self.wp.__version__,
                "gpu": self.torch.cuda.get_device_name(0), "no_new_parameter_fitting": True,
                "release_adapter": "disable controller springs with logK=-inf through upstream eval_springs gate; internal springs unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--mode", choices=("recorded", "lift", "stretch"), default="stretch")
    parser.add_argument("--amplitude", type=float, default=0.05, help="Peak handle displacement in metres")
    parser.add_argument("--max-gaussians", type=int, default=60000)
    parser.add_argument("--no-gaussians", action="store_true")
    parser.add_argument("--release-after", type=int, default=-1, help="Detach both controller groups after this many completed frames; -1 keeps grasp")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    twin = PhysTwinRuntime(args.project_root, args.upstream_root, max_gaussians=args.max_gaussians,
                          enable_gaussians=not args.no_gaussians)
    np = twin.np
    positions, controls = [twin.particles()], [twin.controls()]
    active_states = [twin.active_handles.copy()]
    centers = [twin.gaussians()["centers"]] if twin.gaussian_enabled else []
    t0 = time.perf_counter()
    count = min(args.frames, len(twin.recorded_controls) - 1) if args.mode == "recorded" else args.frames
    for i in range(count):
        if args.release_after >= 0 and i >= args.release_after:
            twin.set_active_handles([False, False])
        if args.mode == "recorded":
            twin.step(controller_positions=twin.recorded_controls[i + 1])
        else:
            # Smooth prescribed *hand* trajectory; plush deforms through the solver.
            alpha = min((i + 1) / max(count * 0.45, 1), 1)
            amount = args.amplitude * (0.5 - 0.5 * np.cos(np.pi * alpha))
            offsets = np.zeros((2, 3), dtype=np.float32)
            if args.mode == "stretch":
                direction = twin.handle_centers[1] - twin.handle_centers[0]
                direction /= np.linalg.norm(direction)
                offsets[0] = -direction * amount
                offsets[1] = direction * amount
            else:
                offsets[:, 2] = -amount
            twin.step(offsets)
        positions.append(twin.particles())
        controls.append(twin.controls())
        active_states.append(twin.active_handles.copy())
        if twin.gaussian_enabled:
            centers.append(twin.gaussians()["centers"])
        if i % 10 == 0:
            print(json.dumps({"frame": i + 1, "of": count, "elapsed_s": time.perf_counter() - t0}), flush=True)
    output = args.output or twin.root / f"outputs/{args.mode}/rollout.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    values = {"positions": np.stack(positions), "controls": np.stack(controls),
              "springs": twin.springs, "spring_stiffness": twin.spring_stiffness,
              "handle_groups": twin.control_groups.astype(np.int32),
              "active_handles": np.stack(active_states),
              "frame_dt": np.array(twin.frame_dt), "num_original": np.array(twin.num_original)}
    if twin.gaussian_enabled:
        values.update(gaussian_centers=np.stack(centers), gaussian_colors=twin.gaussian_colors,
                      gaussian_opacities=twin.gaussian_opacities, gaussian_covariances=twin.gaussian_covariances,
                      gaussian_knn_indices=twin.gaussian_knn_indices)
    np.savez_compressed(output, **values)
    provenance = twin.provenance()
    provenance.update(rollout_mode=args.mode, frames=count + 1, amplitude_m=args.amplitude,
                      release_after_frame=args.release_after,
                      elapsed_s=time.perf_counter() - t0,
                      torch_peak_gpu_gb=twin.torch.cuda.max_memory_allocated() / 1e9,
                      torch_peak_excludes_warp_allocations=True)
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(output), **provenance}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
