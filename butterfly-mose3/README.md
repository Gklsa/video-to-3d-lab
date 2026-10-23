# Butterfly motion with MoSE3

This runs the pretrained [MoSE3 model](https://github.com/mose3-tracker/MoSE3)
on the butterfly clip from the authors' repository. The adapter reduces
GPU memory use, fixes playback when switching colours, and adds MP4, GIF
and image exports.

[RGB video](../media/butterfly/motion_rgb.mp4) ·
[Rigidity-coloured video](../media/butterfly/motion_embedding.mp4)

## Interactive browser viewer

**[Open the hosted browser demo](https://gklsa.github.io/video-to-3d-lab/)**

The public demo replays saved 3D predictions without installing Python or
running GPU inference. Drag to rotate, scroll to zoom, play or scrub the
30 frames, and switch between RGB and rigidity colours. Scrubbing pauses
playback. Open **Controls** for keyboard shortcuts.

## Setup

Prerequisites for local inference:

- Git and Python 3.10 (64-bit).
- An NVIDIA GPU with BF16 support and a compatible NVIDIA driver.
  8 GB VRAM is recommended for the example settings below.

The commands below create an isolated environment and install
[PyTorch 2.2.2 with CUDA 11.8](https://docs.pytorch.org/get-started/previous-versions/)
and the project dependencies. Run them in Windows PowerShell from the folder
where you want to clone the repository.

```powershell
git clone https://github.com/Gklsa/video-to-3d-lab.git
cd video-to-3d-lab/butterfly-mose3

py -3.10 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/mose3-tracker/MoSE3 upstream
git -C upstream checkout c39ef00f7198515d709f40c668688d4f869255ae
python -m pip install --target deps -r requirements-local.txt
python demo.py download
```

The checkpoint is about 6.16 GB. The downloader can resume an interrupted
download and checks the file's SHA256 before use.

## Run

```powershell
python demo.py infer --clip outputs/butterfly_run --frames 30 --width 518 --height 294 --refs 12000 --se3-chunk 1024
python demo.py view --clip outputs/butterfly_run --port 8870
```

For local use, open <http://127.0.0.1:8870/> on the computer running the viewer. You can rotate the
scene, play the sequence and switch between RGB and rigidity colours.
Choose a new output folder when repeating inference; existing runs are kept.

To export the same view as the videos in this repository:

```powershell
python demo.py render --clip outputs/butterfly_run --colors both --query-roi 0.04 0.10 0.87 1.0 --elev 12 --azim -88 --max-points 100000 --width 1920 --height 1080 --fps 10
```

The videos, GIFs and still images will be in `outputs/butterfly_run/rendered/`.

## Results

The run below was recorded on 22 October 2026, using the full butterfly clip.

| Setting | Value |
|---|---|
| Input | 30 frames, 518 × 294, 10 fps |
| SE(3) reference points | 12,000 |
| Fitting batch size | 1,024 |
| Time, including model loading and fitting | 57.3 s |
| Peak allocated GPU memory | 4.72 GB |
| Video export | 1920 × 1080, 10 fps |

The adapter uses BF16 for the backbone and decoder, with selected output
heads in FP32. The exported view shows up to 100,000 predicted points;
1080p is the video size, not the model's input resolution. No ground-truth
tracking accuracy has been measured for this example.

[Run details](../media/butterfly/run_summary.json) include the settings and
file hashes. [Export checks](../media/butterfly/verification.json) record the
decoded video dimensions and frame counts. The earlier, smaller-resolution
run is kept locally for comparison. The memory adaptation is implemented in
[run_low_memory.py](run_low_memory.py).

## Sources

- [MoSE3 project and paper](https://mose3-tracker.github.io/)
- [Official code](https://github.com/mose3-tracker/MoSE3)
- [Pretrained weights](https://huggingface.co/mose3-tracker/MoSE3)

The source clip and pretrained model come from the MoSE3 authors. Their
license terms, including the model's non-commercial conditions, are covered
in [the attribution notes](../THIRD_PARTY_NOTICES.md).
