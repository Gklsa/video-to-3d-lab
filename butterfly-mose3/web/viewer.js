"use strict";

// Draw only the saved MoSE3 samples. Playback selects an existing frame; no
// motion interpolation, geometry generation or model inference happens here.
(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("scene");
  const ui = {
    play: $("play"), frame: $("frame"), time: $("time"), colour: $("colour"),
    size: $("size"), axes: $("axes"), reset: $("reset"), loading: $("loading"),
    loadingText: $("loading-text"), info: $("frame-info"), note: $("data-note"),
  };
  let manifest, gl, renderer, data;
  let frame = 0, playing = true, lastTick = 0, frameClock = 0;
  let dirty = true;
  const camera = { target: [0, 0, 1], distance: 1, azimuth: Math.PI, elevation: 0, fov: .35 };
  let initialCamera;
  const drag = { pointer: null, x: 0, y: 0, pan: false, moved: false };
  let pinch = null;
  const pointers = new Map();

  const subtract = (a, b) => a.map((v, i) => v - b[i]);
  const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
  const norm = (a) => Math.hypot(...a);
  const normalize = (a) => { const n = norm(a) || 1; return a.map((v) => v / n); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  function perspective(fov, aspect, near, far) {
    const f = 1 / Math.tan(fov / 2), a = new Float32Array(16);
    a[0] = f / aspect; a[5] = f; a[10] = (far + near) / (near - far);
    a[11] = -1; a[14] = 2 * far * near / (near - far);
    return a;
  }
  function lookAt(eye, target) {
    const z = normalize(subtract(eye, target));
    const x = normalize(cross([0, -1, 0], z));
    const y = cross(z, x);
    return new Float32Array([
      x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
    ]);
  }
  function multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let v = 0;
      for (let k = 0; k < 4; k++) v += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = v;
    }
    return out;
  }
  function eyePosition() {
    const ce = Math.cos(camera.elevation), se = Math.sin(camera.elevation);
    return [
      camera.target[0] + Math.sin(camera.azimuth) * ce * camera.distance,
      camera.target[1] - se * camera.distance,
      camera.target[2] + Math.cos(camera.azimuth) * ce * camera.distance,
    ];
  }
  function resetCamera() {
    if (!initialCamera) return;
    Object.assign(camera, initialCamera, { target: [...initialCamera.target] });
    dirty = true;
  }
  function showError(error) {
    console.error(error);
    ui.loading.hidden = false;
    ui.loading.classList.add("error");
    ui.loadingText.textContent = error.message || String(error);
  }
  async function loadBinary(path, type, expected) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Could not load ${path} (${response.status}).`);
    const buffer = await response.arrayBuffer();
    const array = new type(buffer);
    if (expected != null && array.length !== expected) throw new Error(`Data size mismatch in ${path}.`);
    return array;
  }

  function shader(type, code) {
    const s = gl.createShader(type);
    gl.shaderSource(s, code); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function program(fragment) {
    const p = gl.createProgram();
    const vertex = `
      attribute vec3 a_position;
      attribute vec3 a_colour;
      attribute float a_visible;
      uniform mat4 u_mvp;
      uniform float u_pointSize;
      uniform float u_distance;
      varying vec3 v_colour;
      void main() {
        vec4 clip = u_mvp * vec4(a_position, 1.0);
        gl_Position = a_visible > 0.5 ? clip : vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = clamp(u_pointSize * u_distance / max(clip.w, 0.001), 1.0, 12.0);
        v_colour = a_colour;
      }`;
    gl.attachShader(p, shader(gl.VERTEX_SHADER, vertex));
    gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fragment)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return {
      program: p, position: gl.getAttribLocation(p, "a_position"), colour: gl.getAttribLocation(p, "a_colour"),
      visible: gl.getAttribLocation(p, "a_visible"), mvp: gl.getUniformLocation(p, "u_mvp"),
      pointSize: gl.getUniformLocation(p, "u_pointSize"), distance: gl.getUniformLocation(p, "u_distance"),
    };
  }
  function buffer(array) {
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW); return b;
  }
  function attribute(b, location, size, type, normalized, offset) {
    if (location < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, type, normalized, 0, offset);
  }
  function makeRenderer() {
    const points = program(`precision mediump float;
      varying vec3 v_colour;
      void main() {
        float r = length(gl_PointCoord - vec2(0.5));
        if (r > 0.5) discard;
        float alpha = 1.0 - smoothstep(0.40, 0.50, r);
        gl_FragColor = vec4(v_colour, alpha);
      }`);
    const result = { points, tracks: buffer(data.tracks), rgb: buffer(data.rgb),
      rigidity: buffer(data.rigidity), visibility: buffer(data.visibility) };
    if (data.axesPositions) {
      result.lines = program(`precision mediump float; varying vec3 v_colour;
        void main(){gl_FragColor=vec4(v_colour,1.0);}`);
      result.axesPositions = buffer(data.axesPositions);
      result.axesColours = buffer(data.axesColours);
      result.axesVisibility = buffer(data.axesVisibility);
    }
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(247 / 255, 248 / 255, 250 / 255, 1);
    return result;
  }
  function draw() {
    if (!renderer || gl.isContextLost()) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const eye = eyePosition();
    // Keep the source image's horizontal framing when an embedded viewer is
    // narrower than the original camera. Only the camera projection changes.
    const aspect = w / h;
    const referenceAspect = manifest.initial_camera?.source_aspect || manifest.width / manifest.height || 1.7619;
    const fitFov = 2 * Math.atan(Math.tan(camera.fov / 2) * Math.max(1, referenceAspect / aspect));
    const matrix = multiply(perspective(fitFov, aspect, Math.max(.001, camera.distance / 500), Math.max(10, camera.distance * 20)), lookAt(eye, camera.target));
    const p = renderer.points;
    gl.useProgram(p.program); gl.uniformMatrix4fv(p.mvp, false, matrix);
    gl.uniform1f(p.pointSize, Number(ui.size.value) * dpr); gl.uniform1f(p.distance, camera.distance);
    attribute(renderer.tracks, p.position, 3, gl.FLOAT, false, frame * manifest.point_count * 3 * 4);
    attribute(renderer[ui.colour.value], p.colour, 3, gl.UNSIGNED_BYTE, true, 0);
    attribute(renderer.visibility, p.visible, 1, gl.UNSIGNED_BYTE, false, frame * manifest.point_count);
    gl.drawArrays(gl.POINTS, 0, manifest.point_count);
    if (renderer.lines && ui.axes.checked) {
      const count = manifest.axis_count * 6, p = renderer.lines;
      gl.useProgram(p.program); gl.uniformMatrix4fv(p.mvp, false, matrix);
      attribute(renderer.axesPositions, p.position, 3, gl.FLOAT, false, frame * count * 3 * 4);
      attribute(renderer.axesColours, p.colour, 3, gl.UNSIGNED_BYTE, true, 0);
      attribute(renderer.axesVisibility, p.visible, 1, gl.UNSIGNED_BYTE, false, frame * count);
      gl.lineWidth(1); gl.drawArrays(gl.LINES, 0, count);
    }
    dirty = false;
  }
  function selectFrame(index) {
    if (!manifest) return;
    frame = clamp(Math.round(index), 0, manifest.frame_count - 1);
    ui.frame.value = frame;
    ui.frame.setAttribute("aria-valuetext", `Frame ${frame + 1} of ${manifest.frame_count}`);
    ui.time.textContent = `${(frame / manifest.fps).toFixed(1)} / ${((manifest.frame_count - 1) / manifest.fps).toFixed(1)} s`;
    ui.info.textContent = `${manifest.point_count.toLocaleString()} points · ${frame + 1} / ${manifest.frame_count}`;
    dirty = true;
  }
  function setPlaying(value) {
    playing = value; frameClock = 0;
    ui.play.textContent = playing ? "Pause" : "Play";
    ui.play.setAttribute("aria-pressed", String(playing));
    ui.play.setAttribute("aria-label", playing ? "Pause playback" : "Play playback");
  }
  function loop(now) {
    if (manifest && renderer) {
      if (lastTick && playing && !document.hidden) {
        frameClock += Math.min((now - lastTick) / 1000, .5);
        const steps = Math.floor(frameClock * manifest.fps);
        if (steps) {
          frameClock -= steps / manifest.fps;
          selectFrame((frame + steps) % manifest.frame_count);
        }
      }
      if (dirty) draw();
    }
    lastTick = now; requestAnimationFrame(loop);
  }
  function prepareAxes(origins, directions, visibility) {
    const T = manifest.frame_count, A = manifest.axis_count, vertices = A * 6;
    data.axesPositions = new Float32Array(T * vertices * 3);
    data.axesVisibility = new Uint8Array(T * vertices);
    data.axesColours = new Uint8Array(vertices * 3);
    const colours = [[204, 74, 67], [59, 153, 101], [61, 111, 194]];
    const length = manifest.axis_length || .018;
    for (let t = 0; t < T; t++) for (let a = 0; a < A; a++) for (let axis = 0; axis < 3; axis++) {
      const n = a * 6 + axis * 2;
      for (let end = 0; end < 2; end++) {
        data.axesVisibility[t * vertices + n + end] = visibility[t * A + a];
        for (let d = 0; d < 3; d++) {
          data.axesPositions[(t * vertices + n + end) * 3 + d] = origins[(t * A + a) * 3 + d] + end * length * directions[((t * A + a) * 3 + axis) * 3 + d];
          if (!t) data.axesColours[(n + end) * 3 + d] = colours[axis][d];
        }
      }
    }
  }
  async function start() {
    gl = canvas.getContext("webgl", { antialias: true, alpha: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error("WebGL is unavailable. Open this viewer in a browser with hardware graphics enabled.");
    const response = await fetch("manifest.json");
    if (!response.ok) throw new Error("The reconstruction data could not be loaded.");
    manifest = await response.json();
    if (!(manifest.frame_count > 0 && manifest.point_count > 0 && manifest.fps > 0)) throw new Error("Invalid reconstruction manifest.");
    const T = manifest.frame_count, N = manifest.point_count, files = manifest.files;
    ui.loadingText.textContent = `Loading ${T} frames of actual predicted points…`;
    const [tracks, rgb, rigidity, visibility] = await Promise.all([
      loadBinary(files.tracks, Float32Array, T * N * 3),
      loadBinary(files.rgb, Uint8Array, N * 3),
      loadBinary(files.rigidity, Uint8Array, N * 3),
      loadBinary(files.visibility, Uint8Array, T * N),
    ]);
    data = { tracks, rgb, rigidity, visibility };
    if (manifest.axis_count && files.axes_origins) {
      const A = manifest.axis_count;
      const values = await Promise.all([
        loadBinary(files.axes_origins, Float32Array, T * A * 3),
        loadBinary(files.axes_directions, Float32Array, T * A * 9),
        loadBinary(files.axes_visible, Uint8Array, T * A),
      ]);
      prepareAxes(...values); $("axes-label").hidden = false;
    }
    const suggested = manifest.camera;
    const eye = suggested?.eye || [0, 0, 0];
    const target = suggested?.target || [0, 0, 1.29];
    const relative = subtract(eye, target), distance = norm(relative);
    initialCamera = { target: [...target], distance, azimuth: Math.atan2(relative[0], relative[2]),
      elevation: Math.asin(clamp(-relative[1] / distance, -.98, .98)), fov: suggested?.vertical_fov_rad || .282 };
    resetCamera(); renderer = makeRenderer();
    ui.frame.max = T - 1;
    for (const control of [ui.play, ui.frame, ui.colour, ui.size]) control.disabled = false;
    ui.note.firstChild.textContent = `${manifest.frame_count} frames · Saved prediction replay · `;
    selectFrame(manifest.query_idx || 0); setPlaying(!window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    ui.loading.hidden = true; dirty = true;
  }

  ui.play.addEventListener("click", () => setPlaying(!playing));
  ui.frame.addEventListener("input", () => { setPlaying(false); selectFrame(Number(ui.frame.value)); frameClock = 0; });
  ui.colour.addEventListener("change", () => { dirty = true; });
  ui.size.addEventListener("input", () => { dirty = true; });
  ui.axes.addEventListener("change", () => { dirty = true; });
  ui.reset.addEventListener("click", resetCamera);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("pointerdown", (event) => {
    if (!initialCamera) return;
    canvas.setPointerCapture(event.pointerId); canvas.focus({ preventScroll: true });
    pointers.set(event.pointerId, [event.clientX, event.clientY]);
    drag.pointer = event.pointerId; drag.x = event.clientX; drag.y = event.clientY;
    drag.pan = event.shiftKey || event.button === 2;
    canvas.classList.add("dragging");
    if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a[0] - b[0], a[1] - b[1]); }
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!initialCamera || !pointers.has(event.pointerId)) return;
    pointers.set(event.pointerId, [event.clientX, event.clientY]);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()], next = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch && next) camera.distance = clamp(camera.distance * pinch / next, initialCamera.distance * .12, initialCamera.distance * 8);
      pinch = next; dirty = true; return;
    }
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    drag.x = event.clientX; drag.y = event.clientY;
    if (drag.pan) {
      const z = normalize(subtract(eyePosition(), camera.target)), x = normalize(cross([0, -1, 0], z)), y = cross(z, x);
      const scale = 2 * camera.distance * Math.tan(camera.fov / 2) / Math.max(1, canvas.clientHeight);
      camera.target = camera.target.map((v, i) => v - x[i] * dx * scale + y[i] * dy * scale);
    } else {
      camera.azimuth -= dx * .007;
      camera.elevation = clamp(camera.elevation + dy * .007, -1.48, 1.48);
    }
    dirty = true;
  });
  function endPointer(event) {
    pointers.delete(event.pointerId); pinch = null;
    if (pointers.size === 1) { const [id, pos] = [...pointers.entries()][0]; drag.pointer = id; [drag.x, drag.y] = pos; }
    if (!pointers.size) { drag.pointer = null; canvas.classList.remove("dragging"); }
  }
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault(); if (!initialCamera) return;
    camera.distance = clamp(camera.distance * Math.exp(clamp(event.deltaY, -150, 150) * .0015), initialCamera.distance * .12, initialCamera.distance * 8);
    dirty = true;
  }, { passive: false });
  canvas.addEventListener("keydown", (event) => {
    if (!initialCamera || !data || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.code === "Space") { event.preventDefault(); setPlaying(!playing); }
    if (event.code === "ArrowLeft" || event.code === "ArrowRight") {
      event.preventDefault(); setPlaying(false);
      selectFrame(frame + (event.code === "ArrowRight" ? 1 : -1));
    }
    if (event.code === "KeyR") resetCamera();
    if (["KeyW", "KeyA", "KeyS", "KeyD"].includes(event.code)) {
      event.preventDefault();
      if (event.code === "KeyA") camera.azimuth -= .08;
      if (event.code === "KeyD") camera.azimuth += .08;
      if (event.code === "KeyW") camera.elevation = clamp(camera.elevation - .08, -1.48, 1.48);
      if (event.code === "KeyS") camera.elevation = clamp(camera.elevation + .08, -1.48, 1.48);
      dirty = true;
    }
    if (["+", "=", "-", "_"].includes(event.key)) {
      event.preventDefault();
      const zoomIn = event.key === "+" || event.key === "=";
      camera.distance = clamp(camera.distance * (zoomIn ? .9 : 1 / .9), initialCamera.distance * .12, initialCamera.distance * 8);
      dirty = true;
    }
  });
  new ResizeObserver(() => { dirty = true; }).observe(canvas);
  canvas.addEventListener("webglcontextlost", (event) => { event.preventDefault(); showError(new Error("The graphics context was interrupted. Reload this viewer to continue.")); });
  canvas.addEventListener("webglcontextrestored", () => { try { renderer = makeRenderer(); ui.loading.hidden = true; dirty = true; } catch (error) { showError(error); } });
  requestAnimationFrame(loop); start().catch(showError);
})();
