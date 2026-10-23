# Third-party sources and result attribution

## MoSE3

This workspace uses [MoSE3](https://github.com/mose3-tracker/MoSE3) at commit
`c39ef00f7198515d709f40c668688d4f869255ae`. The local wrappers load and invoke
that checkout; the upstream source is not bundled. Its BSD 3-Clause license and
additional third-party conditions remain applicable. The upstream license is
retained in [licenses/MoSE3-LICENSE.txt](licenses/MoSE3-LICENSE.txt).

Model weights (including third-party weights) are separately distributed under
CC BY-NC 4.0 according to the authors. The pinned checkpoint and its checksum
are documented in `butterfly-mose3/upstream-source.json`; weights are not in Git.
Do not interpret the code license as a commercial license for the model.

The butterfly RGB source clip is the authors' example, not a personal recording.
The bundled media are local visualizations of inference using that clip and the
authors' pretrained model. They are included with attribution for study and
review; no new ownership or blanket relicense of upstream imagery is claimed.


The static viewer in `butterfly-mose3/web/` replays a fixed subset of 100,000
points from the same saved prediction arrays. Exported coordinates retain
their source float32 values; point size and camera controls only affect display.
`web/source.mp4` and `web/source-frame.png` are the authors' original example
clip and an unchanged first frame, retained with attribution for this study.
The WebGL viewer and portable exporter are local reproduction contributions.

## PhysTwin

The plush uses [PhysTwin](https://github.com/Jianghanxiao/PhysTwin) at commit
`81c718790a37e5e0102eb77af2c6edd34a9db25f` and the authors' prepared
`double_stretch_sloth` sample from dataset revision
`05665bcd0edad8f069c7def9cbb58d2eb75d68d0`.
The original solver, geometry, Gaussian appearance and fitted physical parameters
come from the authors. The upstream checkout and downloaded assets stay outside
Git. File sources and checksums are in
[plush-phystwin/asset_sources.json](plush-phystwin/asset_sources.json).

PhysTwin's main code carries the authors' MIT license, retained in
[licenses/PhysTwin-LICENSE.txt](licenses/PhysTwin-LICENSE.txt). Its bundled
Gaussian-splatting components have additional Inria/MPII research and
noncommercial terms; the upstream copy is retained in
[licenses/Gaussian-Splatting-LICENSE.md](licenses/Gaussian-Splatting-LICENSE.md).
These are separate upstream terms, not a blanket license for this repository
or for all prepared assets. This viewer renders through Viser rather than the
upstream native Gaussian rasterizer.

The included plush video and cover are a new local forward simulation and
rendering of that prepared sample. They are not the authors' demonstration
video. The simulation uses the unchanged official spring-mass solver and
incremental Kabsch/LBS appearance deformation. The local release adapter
disables controller springs through the solver's existing stiffness gate;
internal material springs retain the authors' fitted parameters.

## Local contributions

The local work includes runtime compatibility adapters, interactive viewers,
pull/release controls, video export, documentation and packaging. Both demos
use authors' example assets and pretrained models or fitted parameters.
No new model training, physical parameter fitting, personal toy reconstruction
or full paper benchmark is claimed.
