# Video-to-3D Lab

Butterfly motion playback with MoSE3 and a deformable plush demo with PhysTwin.

## Structure

```text
butterfly-mose3/       Butterfly inference, export and local viewer
  web/                Browser viewer and saved playback data
plush-phystwin/       Plush simulation and local viewer
media/                Demo videos and preview images
START_BUTTERFLY.cmd   Windows launcher for saved butterfly results
START_PLUSH.cmd       Windows launcher for the plush simulation
```

## View the demos

**[Open the public butterfly demo](https://gklsa.github.io/video-to-3d-lab/)**

Click the link to interact in your browser. No installation or local server is needed.

- Drag to rotate; scroll or pinch to zoom; Shift-drag to pan.
- Use **Play/Pause** and the timeline to inspect the motion.
- Switch **Colour** between RGB and rigidity, or enable **Local axes**.
- Use **Reset view** to restore the camera.

**[Watch the plush demo](https://gklsa.github.io/video-to-3d-lab/media/plush/sloth_pull_release.mp4)** for a pull-and-release recording. To manipulate the plush yourself, use the local viewer below.

## Run the projects yourself (optional)

For inference or live simulation, follow the setup guide for your chosen project. The installation commands use Windows PowerShell and require an NVIDIA GPU.

| Project | First use | Launch after setup |
|---|---|---|
| Butterfly | [Set up and run inference](butterfly-mose3/README.md#setup) to generate saved results | Double-click `START_BUTTERFLY.cmd` |
| Plush | [Set up and download the sample](plush-phystwin/README.md#first-setup) | Double-click `START_PLUSH.cmd` |

In the plush viewer, drag the coloured handles or click **Pull both handles**, then **Release grips**. Click **Reset** to start again. **Save current view** and **Save pull-and-release video** write files to `plush-phystwin/outputs/captures/`. Keep the launcher terminal open while using either viewer.

[Sources and licenses](THIRD_PARTY_NOTICES.md)
