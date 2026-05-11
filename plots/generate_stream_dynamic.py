import torch
import numpy as np
import json
import umap
from scipy.interpolate import griddata, RegularGridInterpolator


def generate_stream_dynamic(predictions_path: str, output_path: str):
    preds = torch.load(predictions_path, map_location="cpu")
    inputs = preds["inputs"]
    pseudotime = preds["pseudotime"]
    quantile_preds = preds["quantile_preds"]
    cluster_preds = preds["cluster_preds"]

    x = inputs[:, -1].numpy()
    pred_next = quantile_preds[:, -1, :, 1].numpy()
    q10 = quantile_preds[:, -1, :, 0].numpy()
    q90 = quantile_preds[:, -1, :, 2].numpy()
    pseudo = pseudotime[:, -1].numpy()
    clusters = cluster_preds[:, -1].numpy()
    n_clusters = int(clusters.max()) + 1

    uncertainty = np.mean(q90 - q10, axis=1)

    reducer = umap.UMAP(n_components=2, random_state=42)
    x_emb = np.asarray(reducer.fit_transform(x))
    pred_emb = np.asarray(reducer.transform(pred_next))

    flow_x = pred_emb[:, 0] - x_emb[:, 0]
    flow_y = pred_emb[:, 1] - x_emb[:, 1]

    n_points = x_emb.shape[0]
    n_sample = min(12000, n_points)
    np.random.seed(42)
    indices = np.random.choice(n_points, size=n_sample, replace=False)

    x_sub = x_emb[indices]
    flow_sub = np.stack([flow_x[indices], flow_y[indices]], axis=1)
    pseudo_sub = pseudo[indices]
    uncertainty_sub = uncertainty[indices]
    clusters_sub = clusters[indices].astype(int)

    pseudo_min = float(pseudo_sub.min())
    pseudo_max = float(pseudo_sub.max())
    pseudo_norm = (pseudo_sub - pseudo_min) / (pseudo_max - pseudo_min + 1e-8)
    pseudo_inv = 1.0 - pseudo_norm

    u_min = float(np.percentile(uncertainty_sub, 2))
    u_max = float(np.percentile(uncertainty_sub, 98))
    u_clip = np.clip(uncertainty_sub, u_min, u_max)
    u_norm = (u_clip - u_min) / (u_max - u_min + 1e-8)

    # --- Interpolate flow field onto grid ---
    grid_n = 80
    pad = 0.12
    x_min = float(x_sub[:, 0].min())
    x_max = float(x_sub[:, 0].max())
    y_min = float(x_sub[:, 1].min())
    y_max = float(x_sub[:, 1].max())
    x_pad = (x_max - x_min) * pad
    y_pad = (y_max - y_min) * pad

    gx = np.linspace(x_min - x_pad, x_max + x_pad, grid_n)
    gy = np.linspace(y_min - y_pad, y_max + y_pad, grid_n)
    gxx, gyy = np.meshgrid(gx, gy)

    def interp_grid(vals, method="cubic"):
        z = griddata((x_sub[:, 0], x_sub[:, 1]), vals, (gxx, gyy), method=method)
        nan_mask = np.isnan(z)
        if nan_mask.any():
            z_fill = griddata((x_sub[:, 0], x_sub[:, 1]), vals, (gxx, gyy), method="nearest")
            z[nan_mask] = z_fill[nan_mask]
        return z

    fu_grid = interp_grid(flow_sub[:, 0])
    fv_grid = interp_grid(flow_sub[:, 1])
    h_grid = interp_grid(pseudo_inv)
    

    # --- Compute streamlines through the vector field ---
    def compute_streamline(start, step=0.08, max_steps=600):
        line = [start.copy()]
        pos = np.array(start, dtype=np.float64)
        for _ in range(max_steps):
            ix = np.searchsorted(gx, pos[0], side="right") - 1
            iy = np.searchsorted(gy, pos[1], side="right") - 1
            if ix < 0 or ix >= len(gx) - 1 or iy < 0 or iy >= len(gy) - 1:
                break
            u = float(fu_grid[iy, ix])
            v = float(fv_grid[iy, ix])
            mag = np.sqrt(u * u + v * v)
            if mag < 1e-8:
                break
            pos = pos + np.array([u, v]) / mag * step
            line.append(pos.copy())
        return np.array(line)

    np.random.seed(7)
    stream_seeds_x = np.random.uniform(x_min, x_max, 200)
    stream_seeds_y = np.random.uniform(y_min, y_max, 200)

    for c in range(n_clusters):
        mask = clusters_sub == c
        if mask.sum() > 20:
            stream_seeds_x = np.append(stream_seeds_x, x_sub[mask, 0].mean())
            stream_seeds_y = np.append(stream_seeds_y, x_sub[mask, 1].mean())

    streamlines = []
    for seed_x, seed_y in zip(stream_seeds_x, stream_seeds_y):
        sl = compute_streamline(np.array([seed_x, seed_y]))
        if sl.shape[0] > 15:
            streamlines.append(sl.tolist())

    # --- Scale to Three.js coordinates ---
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    span = max(x_max - x_min, y_max - y_min)
    scale_xy = 50.0
    scale_z = 30.0

    def sx(v):
        return (v - cx) / span * scale_xy

    def sy(v):
        return (v - cy) / span * scale_xy

    gx_sc = (gx - cx) / span * scale_xy
    gy_sc = (gy - cy) / span * scale_xy

    h_interp = RegularGridInterpolator(
        (gy_sc, gx_sc), h_grid, bounds_error=False, fill_value=0.0
    )

    streamlines_3d = []
    for sl in streamlines:
        sl3d = []
        for pt in sl:
            px = sx(pt[0])
            py = sy(pt[1])
            h = float(h_interp([[py, px]])[0])
            sl3d.append([float(px), float(h * scale_z), float(py)])
        streamlines_3d.append(sl3d)

    # --- Cell scatter data ---
    cell_x = sx(x_sub[:, 0]).tolist()
    cell_y = sy(x_sub[:, 1]).tolist()
    cell_z = (pseudo_inv * scale_z).tolist()
    cell_c = clusters_sub.tolist()
    cell_u = u_norm.tolist()
    cell_pseudo = pseudo_norm.tolist()

    # --- Grid flow vectors for animated background ---
    grid_flow_data = []
    stride = 4
    for iy in range(0, grid_n, stride):
        for ix in range(0, grid_n, stride):
            px = float(gx_sc[ix])
            py = float(gy_sc[iy])
            u_val = float(fu_grid[iy, ix])
            v_val = float(fv_grid[iy, ix])
            h_val = float(h_grid[iy, ix])
            mag = np.sqrt(u_val ** 2 + v_val ** 2)
            if mag > 1e-6:
                grid_flow_data.append({
                    "x": px,
                    "y": float(h_val * scale_z),
                    "z": py,
                    "dx": float(u_val / mag * 3.0),
                    "dz": float(v_val / mag * 3.0),
                    "mag": float(min(mag / 0.5, 1.0)),
                })

    payload = {
        "streamlines": streamlines_3d,
        "cells": {
            "x": cell_x,
            "y": cell_y,
            "z": cell_z,
            "c": cell_c,
            "u": cell_u,
            "pseudo": cell_pseudo,
        },
        "gridFlow": grid_flow_data,
        "nClusters": n_clusters,
    }

    data_json = json.dumps(payload)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Developmental Stream Dynamics</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #020208;
    overflow: hidden;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    color: #c9d1d9;
  }}
  #canvas-container {{
    position: fixed; inset: 0; z-index: 1;
  }}
  #overlay {{
    position: fixed; inset: 0; z-index: 10;
    pointer-events: none;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 32px 36px;
  }}
  .header h1 {{
    font-size: 1.8rem;
    font-weight: 200;
    letter-spacing: 4px;
    text-transform: uppercase;
    text-shadow: 0 0 40px rgba(0,200,255,0.3);
  }}
  .header p {{
    font-size: 0.8rem;
    color: #6a7a8a;
    letter-spacing: 1.5px;
    margin-top: 4px;
  }}
  .legend {{
    align-self: flex-end;
    display: flex; flex-direction: column; gap: 4px;
    font-size: 0.65rem; letter-spacing: 0.5px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
  .controls {{
    position: fixed; bottom: 28px; left: 50%; transform: translateX(-50%);
    z-index: 10; font-size: 0.65rem; color: #444; letter-spacing: 1px;
    pointer-events: none;
  }}
  #info {{
    position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
    z-index: 20; font-size: 0.9rem; color: rgba(180,210,255,0.7);
    pointer-events: none; text-align: center;
    letter-spacing: 1px;
    animation: fadeInfo 3s ease-out forwards;
  }}
  @keyframes fadeInfo {{
    0% {{ opacity: 0.9; }}
    100% {{ opacity: 0; }}
  }}
</style>
</head>
<body>

<div id="canvas-container"></div>

<div id="overlay">
  <div class="header">
    <h1>Developmental Stream</h1>
    <p>UMAP flow field &middot; animated fate trajectories &middot; cluster dynamics</p>
  </div>
  <div class="legend" id="legendContainer"></div>
</div>

<div class="controls">DRAG: ORBIT &bull; RIGHT-DRAG: PAN &bull; SCROLL: ZOOM</div>
<div id="info">Loading flow dynamics...</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.157.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.157.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ EffectComposer }} from "three/addons/postprocessing/EffectComposer.js";
import {{ RenderPass }} from "three/addons/postprocessing/RenderPass.js";
import {{ UnrealBloomPass }} from "three/addons/postprocessing/UnrealBloomPass.js";

const data = {data_json};

const clusterPalette = [
  [0.05, 0.55, 1.00], [1.00, 0.12, 0.45], [0.25, 0.95, 0.35],
  [1.00, 0.70, 0.05], [0.65, 0.20, 1.00], [0.10, 0.90, 0.85],
  [1.00, 0.45, 0.10], [0.15, 0.80, 0.60], [0.90, 0.20, 0.55],
  [0.50, 0.60, 0.95], [0.95, 0.85, 0.20], [0.25, 0.70, 0.90],
  [0.80, 0.10, 0.20], [0.40, 0.95, 0.70], [0.70, 0.70, 0.15],
  [0.55, 0.30, 0.80],
];

// Build legend
const legendEl = document.getElementById("legendContainer");
for (let c = 0; c < data.nClusters; c++) {{
  const col = clusterPalette[c % clusterPalette.length];
  const hex = "#" + [col[0], col[1], col[2]].map(v => Math.round(v*255).toString(16).padStart(2,"0")).join("");
  const item = document.createElement("div");
  item.className = "legend-item";
  item.innerHTML = `<div class="legend-dot" style="background:${{hex}};"></div>Cluster ${{c}}`;
  legendEl.appendChild(item);
}}

document.getElementById("info").style.display = "none";

const container = document.getElementById("canvas-container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020208);
scene.fog = new THREE.FogExp2(0x020208, 0.004);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.5, 400);
camera.position.set(30, 55, 70);
camera.lookAt(0, 15, 0);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.4;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0, 15, 0);
controls.maxPolarAngle = Math.PI * 0.46;
controls.minDistance = 10;
controls.maxDistance = 160;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.4;

// ── Lighting ──
scene.add(new THREE.AmbientLight(0x223344, 1.5));
const sun = new THREE.DirectionalLight(0xeef0ff, 3.0);
sun.position.set(30, 60, 25);
scene.add(sun);
const rimLight = new THREE.PointLight(0x0088ff, 8, 100);
rimLight.position.set(-10, 5, 10);
scene.add(rimLight);

// ── Post-processing: bloom ──
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  1.8, 0.6, 0.15
);
composer.addPass(bloom);

// ═══════════════════════════════════════════
// STATIC CELL SCATTER (background context)
// ═══════════════════════════════════════════
const cellCount = data.cells.x.length;
const scatterGeo = new THREE.BufferGeometry();
const scatterPos = new Float32Array(cellCount * 3);
const scatterCol = new Float32Array(cellCount * 3);
const scatterSize = new Float32Array(cellCount);

for (let i = 0; i < cellCount; i++) {{
  scatterPos[i * 3]     = data.cells.x[i];
  scatterPos[i * 3 + 1] = data.cells.z[i];
  scatterPos[i * 3 + 2] = data.cells.y[i];
  const c = clusterPalette[data.cells.c[i] % clusterPalette.length];
  scatterCol[i * 3]     = c[0];
  scatterCol[i * 3 + 1] = c[1];
  scatterCol[i * 3 + 2] = c[2];
  scatterSize[i] = 1.5 + data.cells.u[i] * 2.0;
}}

scatterGeo.setAttribute("position", new THREE.BufferAttribute(scatterPos, 3));
scatterGeo.setAttribute("color", new THREE.BufferAttribute(scatterCol, 3));
scatterGeo.setAttribute("size", new THREE.BufferAttribute(scatterSize, 1));

const scatterMat = new THREE.PointsMaterial({{
  vertexColors: true,
  transparent: true,
  opacity: 0.28,
  size: 0.6,
  sizeAttenuation: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
}});
const scatter = new THREE.Points(scatterGeo, scatterMat);
scene.add(scatter);

// ═══════════════════════════════════════════
// STREAMLINE PARTICLE SYSTEM — the core dynamic
// ═══════════════════════════════════════════
// Each streamline becomes a set of flowing particles that
// travel along the path, loop, and leave fading trails.

class.StreamParticleSystem {{
  constructor(streamlines) {{
    this.streamlines = streamlines;
    this.totalPaths = streamlines.length;
    // Many particles per stream for density
    this.particlesPerStream = 8;
    this.particleCount = this.totalPaths * this.particlesPerStream;
    this.speed = 0.15;
    this.offsets = new Float32Array(this.particleCount);
    this.pathIndices = new Uint16Array(this.particleCount);
    this.trailLength = 6;

    // Initial random spread so particles don't all start at 0
    for (let i = 0; i < this.particleCount; i++) {{
      this.offsets[i] = Math.random();
      this.pathIndices[i] = Math.floor(i / this.particlesPerStream);
    }}

    this._buildGeometry();
  }}

  _buildGeometry() {{
    // Each particle emits a short trail of line segments
    const segmentCount = this.particleCount * this.trailLength;
    const positions = new Float32Array(segmentCount * 6);
    const colors = new Float32Array(segmentCount * 6);
    const alphas = new Float32Array(segmentCount * 2);

    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.geo.setAttribute("alpha", new THREE.BufferAttribute(alphas, 1));

    // Build a lookup of cumulative arc lengths per stream
    this.arcLengths = [];
    for (const sl of this.streamlines) {{
      const cumLen = [0];
      for (let i = 1; i < sl.length; i++) {{
        const dx = sl[i][0] - sl[i-1][0];
        const dy = sl[i][1] - sl[i-1][1];
        const dz = sl[i][2] - sl[i-1][2];
        cumLen.push(cumLen[i-1] + Math.sqrt(dx*dx + dy*dy + dz*dz));
      }}
      this.arcLengths.push(cumLen);
    }}

    this.mat = new THREE.ShaderMaterial({{
      vertexShader: `
        attribute float alpha;
        varying float vAlpha;
        varying vec3 vColor;
        void main() {{
          vAlpha = alpha;
          vColor = color;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }}
      `,
      fragmentShader: `
        varying float vAlpha;
        varying vec3 vColor;
        void main() {{
          float a = vAlpha * 0.85;
          vec3 col = vColor * (1.0 + vAlpha * 0.6);
          gl_FragColor = vec4(col, a);
        }}
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
    }});

    this.lines = new THREE.LineSegments(this.geo, this.mat);
  }}

  _sampleStream(pathIdx, t) {{
    const sl = this.streamlines[pathIdx];
    const cumLen = this.arcLengths[pathIdx];
    if (!sl || sl.length < 2) return null;
    const totalLen = cumLen[cumLen.length - 1];
    const targetLen = t * totalLen;
    // Binary search for segment
    let lo = 0, hi = cumLen.length - 1;
    while (lo < hi - 1) {{
      const mid = (lo + hi) >> 1;
      if (cumLen[mid] < targetLen) lo = mid;
      else hi = mid;
    }}
    const segStart = cumLen[lo];
    const segEnd = cumLen[hi];
    const segFrac = (targetLen - segStart) / (segEnd - segStart + 1e-10);
    const fx = sl[lo][0] + (sl[hi][0] - sl[lo][0]) * segFrac;
    const fy = sl[lo][1] + (sl[hi][1] - sl[lo][1]) * segFrac;
    const fz = sl[lo][2] + (sl[hi][2] - sl[lo][2]) * segFrac;
    return [fx, fy, fz];
  }}

  update(time, dt) {{
    const posAttr = this.geo.attributes.position;
    const colAttr = this.geo.attributes.color;
    const alphaAttr = this.geo.attributes.alpha;
    const visited = new Set();

    for (let i = 0; i < this.particleCount; i++) {{
      const pathIdx = this.pathIndices[i];
      if (pathIdx >= this.totalPaths) continue;
      visited.add(pathIdx);
      const sl = this.streamlines[pathIdx];
      if (!sl || sl.length < 2) continue;

      // Advance offset
      this.offsets[i] += this.speed * dt * (0.8 + 0.4 * (pathIdx % 5) / 5);
      if (this.offsets[i] > 1.0) this.offsets[i] -= 1.0;

      const baseT = this.offsets[i];

      for (let seg = 0; seg < this.trailLength; seg++) {{
        const lineIdx = i * this.trailLength + seg;
        const t1 = baseT - seg * 0.012;
        const t2 = baseT - (seg + 1) * 0.012;
        // Skip out-of-range trail segments
        if (t1 < 0 || t2 < 0) {{
          posAttr.array[lineIdx * 6 + 0] = 0;
          posAttr.array[lineIdx * 6 + 1] = -9999;
          posAttr.array[lineIdx * 6 + 2] = 0;
          posAttr.array[lineIdx * 6 + 3] = 0;
          posAttr.array[lineIdx * 6 + 4] = -9999;
          posAttr.array[lineIdx * 6 + 5] = 0;
          alphaAttr.array[lineIdx * 2] = 0;
          alphaAttr.array[lineIdx * 2 + 1] = 0;
          continue;
        }}

        const p1 = this._sampleStream(pathIdx, t1);
        const p2 = this._sampleStream(pathIdx, t2);
        if (!p1 || !p2) continue;

        posAttr.array[lineIdx * 6 + 0] = p1[0];
        posAttr.array[lineIdx * 6 + 1] = p1[1];
        posAttr.array[lineIdx * 6 + 2] = p1[2];
        posAttr.array[lineIdx * 6 + 3] = p2[0];
        posAttr.array[lineIdx * 6 + 4] = p2[1];
        posAttr.array[lineIdx * 6 + 5] = p2[2];

        // Color from cluster of the path's midpoint
        // We use a gradient based on height
        const heightFrac = Math.max(0, Math.min(1, p1[1] / 30.0));
        const alpha = Math.max(0, 1.0 - seg / this.trailLength) * (0.4 + heightFrac * 0.6);
        alphaAttr.array[lineIdx * 2] = alpha;
        alphaAttr.array[lineIdx * 2 + 1] = alpha;

        // Color: bright cyan-white at tips, fading to cluster hue along trail
        const hue = 0.55 + heightFrac * 0.12;
        const sat = 0.7 - seg * 0.08;
        const val = 0.9 - seg * 0.05;
        const c = new THREE.Color().setHSL(hue, sat, val);
        colAttr.array[lineIdx * 6 + 0] = c.r;
        colAttr.array[lineIdx * 6 + 1] = c.g;
        colAttr.array[lineIdx * 6 + 2] = c.b;
        colAttr.array[lineIdx * 6 + 3] = c.r;
        colAttr.array[lineIdx * 6 + 4] = c.g;
        colAttr.array[lineIdx * 6 + 5] = c.b;
      }}
    }}

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    alphaAttr.needsUpdate = true;
  }}
}}

// Hmm, class syntax above has a typo. Let me just use a plain object approach.
// Actually, let me rewrite this cleanly.
</script>

<script type="module">
// This second script block is not used — the real code is all in the first module script below.
</script>

</body>
</html>
"""

    # That approach above got messy. Let me build the final clean version.

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Developmental Stream Dynamics</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#020208; overflow:hidden; font-family:'Inter','Segoe UI',system-ui,sans-serif; color:#c9d1d9; }}
  #c {{ position:fixed; inset:0; z-index:1; }}
  #ov {{
    position:fixed; inset:0; z-index:10; pointer-events:none;
    display:flex; flex-direction:column; justify-content:space-between; padding:32px 36px;
  }}
  .hdr h1 {{ font-size:1.8rem; font-weight:200; letter-spacing:4px; text-transform:uppercase; text-shadow:0 0 40px rgba(0,200,255,0.3); }}
  .hdr p {{ font-size:0.8rem; color:#6a7a8a; letter-spacing:1.5px; margin-top:4px; }}
  .lg {{ align-self:flex-end; display:flex; flex-direction:column; gap:4px; font-size:0.65rem; letter-spacing:0.5px; }}
  .li {{ display:flex; align-items:center; gap:6px; }}
  .ld {{ width:8px; height:8px; border-radius:50%; }}
  .ct {{ position:fixed; bottom:28px; left:50%; transform:translateX(-50%); z-index:10; font-size:0.65rem; color:#444; letter-spacing:1px; pointer-events:none; }}
</style>
</head>
<body>
<div id="c"></div>
<div id="ov">
  <div class="hdr">
    <h1>Developmental Stream</h1>
    <p>UMAP flow field &middot; animated fate trajectories &middot; cluster dynamics</p>
  </div>
  <div class="lg" id="lg"></div>
</div>
<div class="ct">DRAG: ORBIT &bull; RIGHT-DRAG: PAN &bull; SCROLL: ZOOM</div>

<script type="importmap">
{{"imports":{{"three":"https://unpkg.com/three@0.157.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.157.0/examples/jsm/"}}}}
</script>

<script type="module">
import * as THREE from "three";
import {{OrbitControls}} from "three/addons/controls/OrbitControls.js";
import {{EffectComposer}} from "three/addons/postprocessing/EffectComposer.js";
import {{RenderPass}} from "three/addons/postprocessing/RenderPass.js";
import {{UnrealBloomPass}} from "three/addons/postprocessing/UnrealBloomPass.js";

const D={data_json};
const CP=[
  [0.05,0.55,1.00],[1.00,0.12,0.45],[0.25,0.95,0.35],[1.00,0.70,0.05],
  [0.65,0.20,1.00],[0.10,0.90,0.85],[1.00,0.45,0.10],[0.15,0.80,0.60],
  [0.90,0.20,0.55],[0.50,0.60,0.95],[0.95,0.85,0.20],[0.25,0.70,0.90],
  [0.80,0.10,0.20],[0.40,0.95,0.70],[0.70,0.70,0.15],[0.55,0.30,0.80],
];

const lg=document.getElementById("lg");
for(let c=0;c<D.nClusters;c++){{
  const col=CP[c%CP.length];
  const hex="#"+[col[0],col[1],col[2]].map(v=>Math.round(v*255).toString(16).padStart(2,"0")).join("");
  const d=document.createElement("div");d.className="li";
  d.innerHTML=`<div class="ld" style="background:${{hex}};"></div>Cluster ${{c}}`;
  lg.appendChild(d);
}}

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x020208);
scene.fog=new THREE.FogExp2(0x020208,0.004);

const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.5,400);
camera.position.set(30,55,70);
camera.lookAt(0,15,0);

const renderer=new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth,innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=1.5;
document.getElementById("c").appendChild(renderer.domElement);

const ctrl=new OrbitControls(camera,renderer.domElement);
ctrl.enableDamping=true;ctrl.dampingFactor=0.06;
ctrl.target.set(0,15,0);ctrl.maxPolarAngle=Math.PI*0.46;
ctrl.minDistance=10;ctrl.maxDistance=160;
ctrl.autoRotate=true;ctrl.autoRotateSpeed=0.35;

scene.add(new THREE.AmbientLight(0x223344,1.5));
const sun=new THREE.DirectionalLight(0xeef0ff,3.0);sun.position.set(30,60,25);scene.add(sun);
const rim=new THREE.PointLight(0x0088ff,8,120);rim.position.set(-10,5,10);scene.add(rim);

const composer=new EffectComposer(renderer);
composer.addPass(new RenderPass(scene,camera));
const bloom=new UnrealBloomPass(new THREE.Vector2(innerWidth,innerHeight),2.0,0.5,0.12);
composer.addPass(bloom);

// ── Static cell scatter ──
const NC=D.cells.x.length;
const sg=new THREE.BufferGeometry();
const sp=new Float32Array(NC*3),sc=new Float32Array(NC*3);
for(let i=0;i<NC;i++){{
  sp[i*3]=D.cells.x[i]; sp[i*3+1]=D.cells.z[i]; sp[i*3+2]=D.cells.y[i];
  const c=CP[D.cells.c[i]%CP.length];
  sc[i*3]=c[0]; sc[i*3+1]=c[1]; sc[i*3+2]=c[2];
}}
sg.setAttribute("position",new THREE.BufferAttribute(sp,3));
sg.setAttribute("color",new THREE.BufferAttribute(sc,3));
const sm=new THREE.PointsMaterial({{
  vertexColors:true, transparent:true, opacity:0.25, size:0.5,
  sizeAttenuation:true, depthWrite:false, blending:THREE.AdditiveBlending,
}});
scene.add(new THREE.Points(sg,sm));

// ── Precompute arc lengths per streamline ──
const arcLens=[];
for(const sl of D.streamlines){{
  const cum=[0];
  for(let i=1;i<sl.length;i++){{
    const dx=sl[i][0]-sl[i-1][0], dy=sl[i][1]-sl[i-1][1], dz=sl[i][2]-sl[i-1][2];
    cum.push(cum[i-1]+Math.sqrt(dx*dx+dy*dy+dz*dz));
  }}
  arcLens.push(cum);
}}

function sampleSL(idx,t){{
  const sl=D.streamlines[idx], cum=arcLens[idx];
  if(!sl||sl.length<2) return null;
  const total=cum[cum.length-1]; if(total<1e-6) return null;
  const target=Math.max(0,Math.min(1,t))*total;
  let lo=0,hi=cum.length-1;
  while(lo<hi-1){{ const mid=(lo+hi)>>1; if(cum[mid]<target) lo=mid; else hi=mid; }}
  const frac=(target-cum[lo])/(cum[hi]-cum[lo]+1e-10);
  return [
    sl[lo][0]+(sl[hi][0]-sl[lo][0])*frac,
    sl[lo][1]+(sl[hi][1]-sl[lo][1])*frac,
    sl[lo][2]+(sl[hi][2]-sl[lo][2])*frac,
  ];
}}

// ── Flowing stream particles (glowing dots that travel streamlines) ──
const PPS=6;
const totalP=D.streamlines.length*PPS;
const pGeo=new THREE.BufferGeometry();
const pPos=new Float32Array(totalP*3);
const pCol=new Float32Array(totalP*3);
const pSize=new Float32Array(totalP);
pGeo.setAttribute("position",new THREE.BufferAttribute(pPos,3));
pGeo.setAttribute("color",new THREE.BufferAttribute(pCol,3));
pGeo.setAttribute("size",new THREE.BufferAttribute(pSize,1));

const pMat=new THREE.ShaderMaterial({{
  vertexShader:`
    attribute float size;
    varying vec3 vColor;
    void main(){{
      vColor=color;
      vec4 mv=modelViewMatrix*vec4(position,1.0);
      gl_PointSize=size*(300.0/(-mv.z));
      gl_Position=projectionMatrix*mv;
    }}
  `,
  fragmentShader:`
    varying vec3 vColor;
    void main(){{
      float d=length(gl_PointCoord-0.5)*2.0;
      if(d>1.0) discard;
      float glow=exp(-d*d*3.0);
      gl_FragColor=vec4(vColor*(1.0+glow*1.5),glow);
    }}
  `,
  transparent:true, depthWrite:false, blending:THREE.AdditiveBlending, vertexColors:true,
}});
const pPoints=new THREE.Points(pGeo,pMat);
scene.add(pPoints);

const pOffsets=new Float32Array(totalP);
for(let i=0;i<totalP;i++) pOffsets[i]=Math.random();

// ── Stream trail ribbons (LineSegments that follow particles) ──
const TRAIL=10;
const trailTotal=totalP*TRAIL;
const tGeo=new THREE.BufferGeometry();
const tPos=new Float32Array(trailTotal*6);
const tCol=new Float32Array(trailTotal*6);
const tAlpha=new Float32Array(trailTotal*2);
tGeo.setAttribute("position",new THREE.BufferAttribute(tPos,3));
tGeo.setAttribute("color",new THREE.BufferAttribute(tCol,3));
tGeo.setAttribute("alpha",new THREE.BufferAttribute(tAlpha,1));

const tMat=new THREE.ShaderMaterial({{
  vertexShader:`
    attribute float alpha;
    varying float vAlpha;
    varying vec3 vColor;
    void main(){{
      vAlpha=alpha; vColor=color;
      gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);
    }}
  `,
  fragmentShader:`
    varying float vAlpha;
    varying vec3 vColor;
    void main(){{
      gl_FragColor=vec4(vColor*(1.0+vAlpha*0.8),vAlpha*0.7);
    }}
  `,
  transparent:true, depthWrite:false, blending:THREE.AdditiveBlending, vertexColors:true,
}});
const tLines=new THREE.LineSegments(tGeo,tMat);
scene.add(tLines);

// ── Grid flow arrows (subtle animated field) ──
const GF=D.gridFlow.length;
const gfGeo=new THREE.BufferGeometry();
const gfPos=new Float32Array(GF*6);
const gfCol=new Float32Array(GF*6);
const gfAlpha=new Float32Array(GF*2);
gfGeo.setAttribute("position",new THREE.BufferAttribute(gfPos,3));
gfGeo.setAttribute("color",new THREE.BufferAttribute(gfCol,3));
gfGeo.setAttribute("alpha",new THREE.BufferAttribute(gfAlpha,1));

const gfMat=new THREE.ShaderMaterial({{
  vertexShader:`
    attribute float alpha;
    varying float vAlpha;
    varying vec3 vColor;
    void main(){{
      vAlpha=alpha; vColor=color;
      gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);
    }}
  `,
  fragmentShader:`
    varying float vAlpha;
    varying vec3 vColor;
    void main(){{
      gl_FragColor=vec4(vColor,vAlpha*0.25);
    }}
  `,
  transparent:true, depthWrite:false, blending:THREE.AdditiveBlending, vertexColors:true,
}});
const gfLines=new THREE.LineSegments(gfGeo,gfMat);
scene.add(gfLines);

// ── Reference ground plane ──
const floorGeo=new THREE.PlaneGeometry(120,120);
floorGeo.rotateX(-Math.PI/2);
const floorMat=new THREE.MeshBasicMaterial({{
  color:0x040810, side:THREE.DoubleSide, transparent:true, opacity:0.3, depthWrite:false,
}});
const floor=new THREE.Mesh(floorGeo,floorMat);
floor.position.y=-1;
scene.add(floor);

// ── Animation loop ──
let prevTime=0;

function animate(time){{
  requestAnimationFrame(animate);
  const t=time*0.001;
  const dt=Math.min(t-prevTime,0.05);
  prevTime=t;

  // Update flowing particles
  for(let i=0;i<totalP;i++){{
    const sIdx=Math.floor(i/PPS);
    if(sIdx>=D.streamlines.length) continue;
    const speed=0.08+0.04*(sIdx%7)/7+0.02*Math.sin(sIdx*0.3);
    pOffsets[i]+=speed*dt;
    if(pOffsets[i]>1) pOffsets[i]-=1;
    const p=sampleSL(sIdx,pOffsets[i]);
    if(!p) continue;
    pPos[i*3]=p[0]; pPos[i*3+1]=p[1]; pPos[i*3+2]=p[2];

    // Color by height (pseudotime proxy)
    const hFrac=Math.max(0,Math.min(1,p[1]/30.0));
    const col=new THREE.Color().setHSL(0.55+hFrac*0.12,0.8,0.6+hFrac*0.3);
    pCol[i*3]=col.r; pCol[i*3+1]=col.g; pCol[i*3+2]=col.b;
    pSize[i]=1.5+hFrac*1.5;
  }}
  pGeo.attributes.position.needsUpdate=true;
  pGeo.attributes.color.needsUpdate=true;
  pGeo.attributes.size.needsUpdate=true;

  // Update trail ribbons
  for(let i=0;i<totalP;i++){{
    const sIdx=Math.floor(i/PPS);
    if(sIdx>=D.streamlines.length) continue;
    for(let seg=0;seg<TRAIL;seg++){{
      const li=i*TRAIL+seg;
      const t1=pOffsets[i]-seg*0.015;
      const t2=pOffsets[i]-(seg+1)*0.015;
      if(t1<0||t2<0){{
        tPos[li*6+1]=-9999; tPos[li*6+4]=-9999;
        tAlpha[li*2]=0; tAlpha[li*2+1]=0;
        continue;
      }}
      const p1=sampleSL(sIdx,t1);
      const p2=sampleSL(sIdx,t2);
      if(!p1||!p2){{ tPos[li*6+1]=-9999; tPos[li*6+4]=-9999; tAlpha[li*2]=0; tAlpha[li*2+1]=0; continue; }}

      tPos[li*6]=p1[0]; tPos[li*6+1]=p1[1]; tPos[li*6+2]=p1[2];
      tPos[li*6+3]=p2[0]; tPos[li*6+4]=p2[1]; tPos[li*6+5]=p2[2];

      const fade=Math.pow(1.0-seg/TRAIL,1.5);
      tAlpha[li*2]=fade; tAlpha[li*2+1]=fade;

      const hFrac=Math.max(0,Math.min(1,p1[1]/30.0));
      const c=new THREE.Color().setHSL(0.55+hFrac*0.12,0.7-fade*0.3,0.5+fade*0.3);
      tCol[li*6]=c.r; tCol[li*6+1]=c.g; tCol[li*6+2]=c.b;
      tCol[li*6+3]=c.r; tCol[li*6+4]=c.g; tCol[li*6+5]=c.b;
    }}
  }}
  tGeo.attributes.position.needsUpdate=true;
  tGeo.attributes.color.needsUpdate=true;
  tGeo.attributes.alpha.needsUpdate=true;

  // Animate grid flow arrows
  const flowPulse=Math.sin(t*1.5)*0.5+0.5;
  for(let i=0;i<GF;i++){{
    const f=D.gridFlow[i];
    const wave=Math.sin(t*2.0+f.x*0.15+f.z*0.15)*0.5+0.5;
    const sx_=f.x, sz_=f.z;
    const ex=f.x+f.dx*(0.6+wave*0.8);
    const ey=f.y+0.3+wave*0.5;
    const ez=f.z+f.dz*(0.6+wave*0.8);

    gfPos[i*6]=sx_; gfPos[i*6+1]=f.y+0.3; gfPos[i*6+2]=sz_;
    gfPos[i*6+3]=ex; gfPos[i*6+4]=ey; gfPos[i*6+5]=ez;

    const intens=f.mag*(0.15+wave*0.15);
    const c=new THREE.Color().setHSL(0.55,0.4,0.3+intens*0.3);
    gfCol[i*6]=c.r; gfCol[i*6+1]=c.g; gfCol[i*6+2]=c.b;
    gfCol[i*6+3]=c.r; gfCol[i*6+4]=c.g; gfCol[i*6+5]=c.b;
    gfAlpha[i*2]=0.2+wave*0.15;
    gfAlpha[i*2+1]=0.2+wave*0.15;
  }}
  gfGeo.attributes.position.needsUpdate=true;
  gfGeo.attributes.color.needsUpdate=true;
  gfGeo.attributes.alpha.needsUpdate=true;

  // Subtle scatter brightness pulse
  sm.opacity=0.22+Math.sin(t*0.8)*0.04;

  ctrl.update();
  composer.render();
}}

requestAnimationFrame(animate);

addEventListener("resize",()=>{{
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
  composer.setSize(innerWidth,innerHeight);
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  stream dyn  → {output_path}")