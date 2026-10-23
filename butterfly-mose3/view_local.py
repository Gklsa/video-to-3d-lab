"""Launch the authors' viewer on loopback with explicit experiment attribution."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deps"))
sys.path.insert(0, str(ROOT / "upstream"))
import viser

original_server = viser.ViserServer


def local_server(*args, **kwargs):
    kwargs.setdefault("host", "127.0.0.1")
    kwargs.setdefault("label", "MoSE3 | 3D motion study")
    return original_server(*args, **kwargs)


if __name__ == "__main__":
    viser.ViserServer = local_server
    path = ROOT / "upstream/visualize_viser.py"
    source = path.read_text(encoding="utf-8")
    # The reference viewer removes/recreates point-cloud handles on a colour
    # change. Concurrent playback can then assign visibility to a removed
    # handle and terminate the playback thread (observed with viser 1.1.1).
    # Recolour the existing handles instead; points and frame IDs are unchanged.
    old = '''    def upload_points() -> None:
        for h in layers["points"]:
            h.remove()
        colors = s["rgb"] if g_color.value == "RGB" else s["pca"]
'''
    new = '''    def upload_points() -> None:
        colors = s["rgb"] if g_color.value == "RGB" else s["pca"]
        if layers["points"]:
            for handle in layers["points"]:
                handle.colors = colors
                handle.point_size = float(g_psize.value)
            return
'''
    if source.count(old) != 1:
        raise RuntimeError("The pinned viewer source changed; inspect the colour-update adapter")
    source = source.replace(old, new, 1)
    # Refined display: dense small circular points with accurate stored
    # coordinates. This changes only rendering, never the prediction arrays.
    source = source.replace("point_size = 0.004 * depth", "point_size = 0.0015 * depth", 1)
    source = source.replace('initial_value=8.0, min=0.5',
                            'initial_value=10.0, min=0.5', 1)
    source = source.replace(
        'point_size=float(g_psize.value), visible=False)',
        'point_size=float(g_psize.value), point_shape="circle", '
        'precision="float32", point_shading="flat", visible=False)', 1)
    source = source.replace(
        'point_size=head_size, visible=False)',
        'point_size=head_size, point_shape="circle", '
        'precision="float32", point_shading="flat", visible=False)', 1)
    caption_at = '    H, W = z["tracks"].shape[1:3]\n'
    if source.count(caption_at) != 1:
        raise RuntimeError("The pinned viewer input-shape caption needs inspection")
    source = source.replace(caption_at, caption_at +
        '    server.gui.add_markdown(f"**{W} × {H} input · {T} frames**  \\n'
        'Local pretrained inference · authors’ example video")\n', 1)
    exec(compile(source, str(path), "exec"),
         {"__name__": "__main__", "__file__": str(path)})
