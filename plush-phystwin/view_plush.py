"""Interactive PhysTwin viewer: official physical solver, browser Gaussian display.

GUI callbacks only queue inputs. The main thread owns every call into Warp and
PyTorch, including initialization, reset, controller release and forward steps.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import webbrowser


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--deps", type=Path, action="append", default=[])
    parser.add_argument("--case-name", default="double_stretch_sloth")
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--fps", type=float, default=12.0, help="Maximum displayed physics frames per second")
    parser.add_argument("--max-gaussians", type=int, default=60000)
    parser.add_argument("--browser", dest="browser", action="store_true", help="Open the loopback viewer in a browser")
    parser.add_argument("--no-browser", dest="browser", action="store_false")
    parser.set_defaults(browser=False)
    return parser.parse_args()


@dataclass
class InputState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    target_offsets: object = None
    paused: bool = False
    reset_requested: bool = False
    release_requested: bool = False
    preset_requested: str | None = None
    pull_cm: float = 5.0
    show_particles: bool = False
    show_gaussians: bool = True
    show_handles: bool = True
    view_requested: bool = False
    record_requested: bool = False
    recording: bool = False
    capture_requested: bool = False
    error: str | None = None


def main():
    args = parse_args()
    if args.fps <= 0 or args.fps > 60:
        raise ValueError("--fps must be in (0, 60]")
    root = args.project_root.resolve()
    # Package directories remain separate from existing Conda environments.
    for deps in (root / "deps", *args.deps):
        if deps.exists():
            sys.path.insert(0, str(deps.resolve()))
    import numpy as np
    import viser
    from physics_runner import PhysTwinRuntime

    print("Loading the authors' PhysTwin sloth and trained physical parameters...", flush=True)
    twin = PhysTwinRuntime(root, args.upstream_root, case_name=args.case_name,
                          max_gaussians=args.max_gaussians)
    initial_particles = np.asarray(twin.particles(), dtype=np.float32)
    initial_handles = np.asarray(twin.handle_centers, dtype=np.float32).copy()
    if initial_handles.shape != (2, 3):
        raise ValueError("This viewer expects two PhysTwin control groups")
    state = InputState(target_offsets=np.zeros((2, 3), dtype=np.float32))
    actual_offsets = np.zeros((2, 3), dtype=np.float32)
    active_handles = np.ones(2, dtype=bool)
    center = (np.percentile(initial_particles, 2, axis=0) + np.percentile(initial_particles, 98, axis=0)) / 2
    radius = max(float(np.max(np.ptp(initial_particles, axis=0))), 0.25)

    server = viser.ViserServer(host="127.0.0.1", port=args.port,
                               label="PhysTwin | Deformable plush")
    server.gui.configure_theme(dark_mode=False, brand_color=(28, 88, 130),
                               control_width="medium", show_logo=False,
                               show_share_button=False)
    server.scene.set_up_direction(getattr(twin, "up_direction", "-z"))
    server.scene.world_axes.visible = False
    server.scene.add_grid("/table", width=1.2, height=1.2, plane="xy",
                          cell_size=0.05, section_size=0.25,
                          cell_color=(226, 232, 237), section_color=(188, 201, 213),
                          plane_color=(249, 250, 251), plane_opacity=0.6,
                          position=(float(center[0]), float(center[1]), 0.002))

    gaussians = twin.gaussians()
    if not isinstance(gaussians, dict):
        raise RuntimeError("The runtime must expose centers, covariances, rgbs and opacities")
    splats = server.scene.add_gaussian_splats("/plush/appearance", **gaussians)
    # A separate diagnostic layer shows simulated particles without hiding
    # the fact that the visible Gaussian surface uses an appearance model.
    particle_colors = np.tile(np.array([0.16, 0.45, 0.68], np.float32), (len(initial_particles), 1))
    point_cloud = server.scene.add_point_cloud("/plush/particles", initial_particles,
                                               particle_colors, point_size=0.002,
                                               point_shape="circle", precision="float32",
                                               visible=False)

    handles = []
    labels = []
    for i in range(2):
        handle = server.scene.add_transform_controls(
            f"/handles/hand_{i + 1}", scale=0.085, line_width=2.0,
            disable_rotations=True, disable_sliders=True, depth_test=False,
            position=tuple(initial_handles[i]),
        )
        handles.append(handle)
        labels.append(server.scene.add_label(f"/handles/label_{i + 1}", f"Handle {i + 1}",
                               position=tuple(initial_handles[i] + [0.0, 0.0, -0.025]),
                               font_scene_height=0.016, font_size_mode="scene"))

        def on_drag(event, index=i):
            offset = np.asarray(event.target.position, dtype=np.float32) - initial_handles[index]
            with state.lock:
                if state.recording:
                    return
                state.target_offsets[index] = np.clip(offset, -0.15, 0.15)
                state.preset_requested = None

        handle.on_update(on_drag)

    server.gui.add_markdown("## Deformable plush\nDrag a handle, pull both hands apart, or release the grips.\n\n"
                            "The toy moves through the **PhysTwin physics solver**. The browser displays the authors' Gaussian appearance.")
    pause = server.gui.add_checkbox("Pause simulation", initial_value=False)
    pull_distance = server.gui.add_slider("Pull distance (cm)", min=1.0, max=10.0, step=0.5, initial_value=5.0)
    pull = server.gui.add_button("Pull both handles")
    release = server.gui.add_button("Release grips")
    reset = server.gui.add_button("Reset")
    return_handles = server.gui.add_button("Return handles to start")
    with server.gui.add_folder("View"):
        show_surface = server.gui.add_checkbox("Gaussian appearance", initial_value=True)
        show_particles = server.gui.add_checkbox("Simulation particles", initial_value=False)
        show_handles = server.gui.add_checkbox("Control handles", initial_value=True)
        reset_view = server.gui.add_button("Reset camera")
    with server.gui.add_folder("Save"):
        capture = server.gui.add_button("Save current view")
        record = server.gui.add_button("Save pull-and-release video")
        server.gui.add_markdown("Exports go to `outputs/captures`. Video records 90 actual physics steps, with the camera view shown here.")
    status = server.gui.add_markdown("Loading scene...")
    server.gui.add_markdown("**Source**\n[PhysTwin · ICCV 2025](https://github.com/Jianghanxiao/PhysTwin)\n\n"
                            "Prepared `double_stretch_sloth` example and trained parameters from the authors. "
                            "This local run adds interactive controls and browser rendering; it does not fit a new toy from video.")

    @pause.on_update
    def on_pause(event):
        with state.lock:
            state.paused = bool(event.target.value)

    @pull_distance.on_update
    def on_pull_distance(event):
        with state.lock:
            state.pull_cm = float(event.target.value)

    @pull.on_click
    def on_pull(_):
        with state.lock:
            if not state.recording:
                state.preset_requested = "pull"

    @return_handles.on_click
    def on_return(_):
        with state.lock:
            if not state.recording:
                state.preset_requested = "return"

    @release.on_click
    def on_release(_):
        with state.lock:
            if not state.recording:
                state.release_requested = True

    @reset.on_click
    def on_reset(_):
        with state.lock:
            state.reset_requested = True

    @show_surface.on_update
    def on_surface(event):
        with state.lock:
            state.show_gaussians = bool(event.target.value)

    @show_particles.on_update
    def on_particles(event):
        with state.lock:
            state.show_particles = bool(event.target.value)

    @show_handles.on_update
    def on_handles(event):
        with state.lock:
            state.show_handles = bool(event.target.value)

    @reset_view.on_click
    def on_view(_):
        with state.lock:
            state.view_requested = True

    @capture.on_click
    def on_capture(_):
        with state.lock:
            state.capture_requested = True

    @record.on_click
    def on_record(_):
        with state.lock:
            if not state.recording:
                state.record_requested = True

    def set_camera(client):
        client.camera.position = center + np.array([-1.05, -0.2, -1.30]) * radius
        client.camera.look_at = center
        client.camera.up_direction = (0.0, 0.0, -1.0)
        client.camera.fov = np.deg2rad(45.0)

    server.on_client_connect(set_camera)
    url = f"http://127.0.0.1:{server.get_port()}/"
    print(f"PhysTwin ready: {url}", flush=True)
    if args.browser:
        webbrowser.open(url)

    last_status = 0.0
    measured_fps = 0.0
    frame_count = 0
    fps_window = time.perf_counter()
    error = None
    recorder = None
    saved_message = ""

    def capture_size(client):
        width = max(16, min(1280, int(client.camera.image_width)) // 16 * 16)
        height = max(16, min(720, int(client.camera.image_height), int(width * 9 / 16)) // 16 * 16)
        return height, width

    def close_recording(completed):
        nonlocal recorder, saved_message
        if recorder is None:
            return
        recorder["writer"].close()
        metadata = twin.provenance()
        metadata.update(video=str(recorder["path"]), frames=recorder["frame"],
                        completed=bool(completed), video_fps=1.0 / twin.frame_dt,
                        preset="5cm symmetric controller pull, then actual grip-spring release",
                        appearance_renderer="Viser browser Gaussian splats",
                        camera_position=recorder["client"].camera.position.tolist(),
                        camera_wxyz=recorder["client"].camera.wxyz.tolist(),
                        controller_release_step=60)
        recorder["path"].with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        saved_message = f"Saved: `{recorder['path'].name}` ({recorder['frame']} frames)"
        print(saved_message, flush=True)
        recorder = None
        with state.lock:
            state.recording = False

    try:
        while True:
            started = time.perf_counter()
            with state.lock:
                reset_now, release_now = state.reset_requested, state.release_requested
                preset, pull_cm = state.preset_requested, state.pull_cm
                view_now = state.view_requested
                record_now, capture_now = state.record_requested, state.capture_requested
                state.reset_requested = state.release_requested = state.view_requested = False
                state.record_requested = state.capture_requested = False
                state.preset_requested = None
                paused = state.paused
                draw_surface, draw_particles, draw_handles = state.show_gaussians, state.show_particles, state.show_handles

            if reset_now:
                close_recording(False)
                twin.reset()
                actual_offsets.fill(0)
                active_handles[:] = True
                error = None
                with state.lock:
                    state.target_offsets[:] = 0
                    state.paused = False
                pause.value = False
                paused = False
                for i, handle in enumerate(handles):
                    handle.position = initial_handles[i]

            if release_now:
                # This disables the real controller springs, rather than
                # assigning a precomputed falling trajectory to the object.
                active_handles[:] = False
                twin.set_active_handles(active_handles)
                with state.lock:
                    state.paused = False
                pause.value = False
                paused = False
            if preset is not None and active_handles.all():
                goal = np.zeros((2, 3), dtype=np.float32)
                if preset == "pull":
                    direction = initial_handles[1] - initial_handles[0]
                    direction /= np.linalg.norm(direction)
                    goal[0], goal[1] = -direction * pull_cm / 100.0, direction * pull_cm / 100.0
                with state.lock:
                    state.target_offsets[:] = goal
                    state.paused = False
                pause.value = False
                paused = False
                for i, handle in enumerate(handles):
                    handle.position = initial_handles[i] + goal[i]

            if view_now:
                for client in server.get_clients().values():
                    set_camera(client)

            if record_now and recorder is None and server.get_clients():
                import imageio.v2 as imageio
                twin.reset()
                actual_offsets.fill(0)
                active_handles[:] = True
                error = None
                with state.lock:
                    state.target_offsets[:] = 0
                    state.paused = False
                paused = False
                pause.value = False
                folder = root / "outputs/captures"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"sloth_pull_release_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                recorder = {"path": path, "frame": 0, "client": next(iter(server.get_clients().values())),
                            "writer": imageio.get_writer(path, fps=1.0 / twin.frame_dt, codec="libx264", quality=8,
                                                         macro_block_size=16, ffmpeg_log_level="error")}
                with state.lock:
                    state.recording = True
                for handle in handles:
                    handle.fixed = True

            if recorder is not None:
                phase = recorder["frame"]
                goal = np.zeros((2, 3), dtype=np.float32)
                direction = initial_handles[1] - initial_handles[0]
                direction /= np.linalg.norm(direction)
                # The preset moves only the kinematic hand controls. Every
                # plush state is generated by the physical solver below.
                amount = 0.05 * min(phase / 40.0, 1.0)
                goal[0], goal[1] = -direction * amount, direction * amount
                with state.lock:
                    state.target_offsets[:] = goal
                if phase >= 60:
                    active_handles[:] = False
                    twin.set_active_handles(active_handles)
                for i, handle in enumerate(handles):
                    handle.position = initial_handles[i] + goal[i]

            if server.get_clients() and not paused and error is None:
                with state.lock:
                    target = np.asarray(state.target_offsets).copy()
                # A mouse drag can jump several centimetres in one browser
                # event. Limit controller displacement per physical frame.
                delta = target - actual_offsets
                lengths = np.linalg.norm(delta, axis=1, keepdims=True)
                actual_offsets += delta * np.minimum(1.0, 0.0025 / np.maximum(lengths, 1e-9))
                try:
                    twin.step(actual_offsets, active_handles=active_handles)
                    values = twin.gaussians()
                    with server.atomic():
                        splats.set_gaussians(**values)
                        if draw_particles:
                            point_cloud.points = np.asarray(twin.particles(), dtype=np.float32)
                    frame_count += 1
                    if recorder is not None:
                        client = recorder["client"]
                        height, width = capture_size(client)
                        server.flush()
                        frame = client.camera.get_render(height=height, width=width, transport_format="jpeg", timeout=20.0)
                        recorder["writer"].append_data(frame[..., :3])
                        if recorder["frame"] == 0:
                            import imageio.v2 as imageio
                            imageio.imwrite(recorder["path"].with_suffix(".png"), frame[..., :3])
                        recorder["frame"] += 1
                        if recorder["frame"] >= 90:
                            close_recording(True)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    print(traceback.format_exc(), flush=True)
                    with state.lock:
                        state.paused = True
                    pause.value = True
                    close_recording(False)

            if capture_now and server.get_clients():
                import imageio.v2 as imageio
                client = next(iter(server.get_clients().values()))
                height, width = capture_size(client)
                server.flush()
                frame = client.camera.get_render(height=height, width=width, transport_format="jpeg", timeout=20.0)
                folder = root / "outputs/captures"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"sloth_view_{datetime.now():%Y%m%d_%H%M%S}.png"
                imageio.imwrite(path, frame[..., :3])
                saved_message = f"Saved: `{path.name}`"
                print(saved_message, flush=True)

            splats.visible = draw_surface
            point_cloud.visible = draw_particles
            for i, handle in enumerate(handles):
                handle.fixed = recorder is not None
                handle.visible = draw_handles and bool(active_handles[i])
                labels[i].visible = handle.visible
                labels[i].position = initial_handles[i] + actual_offsets[i] + [0.0, 0.0, -0.025]
            pull.disabled = return_handles.disabled = release.disabled = (recorder is not None or not bool(active_handles.all()))
            record.disabled = recorder is not None
            now = time.perf_counter()
            if now - fps_window >= 1.0:
                measured_fps = frame_count / (now - fps_window)
                frame_count = 0
                fps_window = now
            if now - last_status >= 0.5:
                sim_time = float(getattr(twin, "simulation_time", twin.step_count * twin.frame_dt))
                state_text = "Paused" if paused else "Running"
                grips = "Holding" if active_handles.all() else "Released · Reset to grip again"
                status.content = (f"**{state_text} · {grips}**  \n"
                                  f"Simulation: {sim_time:.2f} s · step {twin.frame_dt:.4f} s  \n"
                                  f"Display: {measured_fps:.1f} fps · {len(gaussians['centers']):,} Gaussians")
                if error:
                    status.content += f"\n\n**Simulation stopped.** {error}\n\nUse Reset to restore the initial state."
                if recorder is not None:
                    status.content += f"\n\nRecording: {recorder['frame']} / 90 frames"
                elif saved_message:
                    status.content += f"\n\n{saved_message}"
                last_status = now
            time.sleep(max(0.001, 1.0 / args.fps - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        print("PhysTwin viewer stopped.", flush=True)
    finally:
        close_recording(False)
        server.stop()


if __name__ == "__main__":
    main()
