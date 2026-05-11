import torch
import numpy as np
import json
import umap
from scipy.interpolate import griddata, RegularGridInterpolator
import matplotlib.cm as cm


def generate_terrain_3d(predictions_path: str, output_path: str):
    preds = torch.load(predictions_path, map_location="cpu")
    inputs = preds["inputs"]
    pseudotime = preds["pseudotime"]
    quantile_preds = preds["quantile_preds"]
    cluster_preds = preds["cluster_preds"]

    x: np.ndarray = inputs[:, -1].numpy()
    pred_next: np.ndarray = quantile_preds[:, -1, :, 1].numpy()
    q10: np.ndarray = quantile_preds[:, -1, :, 0].numpy()
    q90: np.ndarray = quantile_preds[:, -1, :, 2].numpy()
    pseudo: np.ndarray = pseudotime[:, -1].numpy()
    clusters: np.ndarray = cluster_preds[:, -1].numpy()
    n_clusters = int(clusters.max()) + 1

    uncertainty: np.ndarray = np.mean(q90 - q10, axis=1)

    reducer = umap.UMAP(n_components=2, random_state=42)
    x_emb = np.asarray(reducer.fit_transform(x))
    pred_emb = np.asarray(reducer.transform(pred_next))

    flow_x: np.ndarray = pred_emb[:, 0] - x_emb[:, 0]
    flow_y: np.ndarray = pred_emb[:, 1] - x_emb[:, 1]

    n_points = x_emb.shape[0]
    n_sample = min(10000, n_points)
    np.random.seed(42)
    indices = np.random.choice(n_points, size=n_sample, replace=False)

    x_sub = x_emb[indices]
    flow_x_sub = flow_x[indices]
    flow_y_sub = flow_y[indices]
    pseudo_sub = pseudo[indices]
    uncertainty_sub = uncertainty[indices]
    clusters_sub = clusters[indices].astype(int)

    pseudo_min = float(pseudo_sub.min())
    pseudo_max = float(pseudo_sub.max())
    pseudo_norm = (pseudo_sub - pseudo_min) / (pseudo_max - pseudo_min + 1e-8)
    pseudo_inv = 1.0 - pseudo_norm  # invert: top = stem cells, bottom = differentiated

    u_min = float(np.percentile(uncertainty_sub, 2))
    u_max = float(np.percentile(uncertainty_sub, 98))
    u_clip = np.clip(uncertainty_sub, u_min, u_max)
    u_norm = (u_clip - u_min) / (u_max - u_min + 1e-8)

    # --- Interpolate terrain grid ---
    grid_n = 80
    pad = 0.08
    x_min = float(x_sub[:, 0].min())
    x_max = float(x_sub[:, 0].max())
    y_min = float(x_sub[:, 1].min())
    y_max = float(x_sub[:, 1].max())
    x_pad = (x_max - x_min) * pad
    y_pad = (y_max - y_min) * pad

    gx = np.linspace(x_min - x_pad, x_max + x_pad, grid_n)
    gy = np.linspace(y_min - y_pad, y_max + y_pad, grid_n)
    gxx, gyy = np.meshgrid(gx, gy)

    def interp_to_grid(vals, px, py, method="cubic"):
        z = griddata((px, py), vals, (gxx, gyy), method=method)
        nan_mask = np.isnan(z)
        if nan_mask.any() and method == "cubic":
            z_fill = griddata((px, py), vals, (gxx, gyy), method="nearest")
            z[nan_mask] = z_fill[nan_mask]
        return z

    h_grid = interp_to_grid(pseudo_inv, x_sub[:, 0], x_sub[:, 1])
    u_grid = np.clip(interp_to_grid(u_norm, x_sub[:, 0], x_sub[:, 1]), 0.0, 1.0)
    fu_grid = interp_to_grid(flow_x_sub, x_sub[:, 0], x_sub[:, 1])
    fv_grid = interp_to_grid(flow_y_sub, x_sub[:, 0], x_sub[:, 1])

    # --- Compute streamlines ---
    def compute_streamline(start, step=0.06, max_steps=400):
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

    np.random.seed(123)
    n_seeds = 50
    seed_x = np.random.uniform(x_min, x_max, n_seeds)
    seed_y = np.random.uniform(y_min, y_max, n_seeds)

    for c in range(n_clusters):
        mask = clusters_sub == c
        if mask.sum() > 30:
            seed_x = np.append(seed_x, x_sub[mask, 0].mean())
            seed_y = np.append(seed_y, x_sub[mask, 1].mean())

    streamlines_2d = []
    for sx, sy in zip(seed_x, seed_y):
        sl = compute_streamline(np.array([sx, sy]))
        if sl.shape[0] > 8:
            streamlines_2d.append(sl.tolist())

    # --- Scale and center ---
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    span = max(x_max - x_min, y_max - y_min)

    scale_xy = 40.0
    scale_z = 30.0

    def scale_x(v):
        return (v - cx) / span * scale_xy

    def scale_y(v):
        return (v - cy) / span * scale_xy

    gx_sc = scale_x(gx)
    gy_sc = scale_y(gy)
    gxx_sc, gyy_sc = np.meshgrid(gx_sc, gy_sc)

    sx_sc = scale_x(x_sub[:, 0])
    sy_sc = scale_y(x_sub[:, 1])
    sz_sc = pseudo_inv * scale_z

    h_interp = RegularGridInterpolator(
        (gy_sc, gx_sc), h_grid, bounds_error=False, fill_value=0.0
    )

    streamlines_3d = []
    for sl in streamlines_2d:
        sl3d = []
        for pt in sl:
            sx = scale_x(pt[0])
            sy = scale_y(pt[1])
            h = float(h_interp([[sy, sx]])[0]) + 0.6
            sl3d.append([float(sx), float(h), float(sy)])
        streamlines_3d.append(sl3d)

    # Magma colors — precompute RGB for each terrain vertex
    magma = cm.get_cmap("magma")
    u_flat = u_grid.ravel()
    u_colors = (magma(u_flat)[:, :3]).tolist()

    payload = {
        "nx": grid_n,
        "ny": grid_n,
        "gx": gx_sc.tolist(),
        "gy": gy_sc.tolist(),
        "heights": (h_grid.ravel() * scale_z).tolist(),
        "u_colors": u_colors,
        "streamlines": streamlines_3d,
        "points": [
            {
                "x": float(sx_sc[i]),
                "y": float(sy_sc[i]),
                "z": float(sz_sc[i]),
                "c": int(clusters_sub[i]),
                "u": float(u_norm[i]),
            }
            for i in range(n_sample)
        ],
        "n_clusters": n_clusters,
    }

    data_json = json.dumps(payload)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Waddington Landscape · Terrain View</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #040508;
    overflow: hidden;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    color: #c9d1d9;
  }}
  #canvas-container {{
    position: fixed; inset: 0;
    z-index: 1;
  }}
  #overlay {{
    position: fixed; inset: 0;
    z-index: 10;
    pointer-events: none;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 36px 40px;
  }}
  .header h1 {{
    font-size: 2rem;
    font-weight: 300;
    letter-spacing: 3px;
    text-transform: uppercase;
    text-shadow: 0 0 40px rgba(120,160,255,0.5);
  }}
  .header p {{
    font-size: 0.85rem;
    color: #8899aa;
    letter-spacing: 1.5px;
    margin-top: 6px;
  }}
  .legend {{
    align-self: flex-end;
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 0.7rem;
    letter-spacing: 0.5px;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 8px;
  }}
  .legend-swatch {{
    width: 10px; height: 10px; border-radius: 2px;
  }}
  #magma-bar {{
    width: 12px; height: 80px;
    border-radius: 2px;
    margin-top: 10px;
  }}
  .controls {{
    position: fixed;
    bottom: 36px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 10;
    font-size: 0.7rem;
    color: #555;
    letter-spacing: 1px;
    pointer-events: none;
  }}
  #tooltip {{
    position: fixed;
    z-index: 20;
    pointer-events: none;
    background: rgba(8,10,14,0.92);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.75rem;
    display: none;
    backdrop-filter: blur(8px);
  }}
</style>
</head>
<body>

<div id="canvas-container"></div>

<div id="overlay">
  <div class="header">
    <h1>Developmental Landscape</h1>
    <p>UMAP terrain · height = pseudotime · color = uncertainty · streams = fate flow</p>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#ffcc00;"></div> High uncertainty</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#000004;"></div> Low uncertainty</div>
    <div id="magma-bar"></div>
  </div>
</div>

<div class="controls">LEFT-DRAG: ORBIT &bull; RIGHT-DRAG: PAN &bull; SCROLL: ZOOM</div>
<div id="tooltip"></div>

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

const data = {data_json};

const container = document.getElementById("canvas-container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x040508);
scene.fog = new THREE.FogExp2(0x040508, 0.0006);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 500);
camera.position.set(35, 42, 50);
camera.lookAt(0, 18, 0);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = false;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 18, 0);
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 8;
controls.maxDistance = 180;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.25;
controls.update();

// ── Lighting ──
const ambient = new THREE.AmbientLight(0x334466, 1.2);
scene.add(ambient);

const sun = new THREE.DirectionalLight(0xffeedd, 3.5);
sun.position.set(40, 60, 20);
scene.add(sun);

const fill = new THREE.DirectionalLight(0x4488cc, 1.0);
fill.position.set(-30, 10, -40);
scene.add(fill);

const rim = new THREE.PointLight(0x0066cc, 12, 120);
rim.position.set(0, 5, 0);
scene.add(rim);

// ── Magma colorbar canvas ──
(function drawMagmaBar() {{
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 128, 0, 0);
  gradient.addColorStop(0.00, "#000004");
  gradient.addColorStop(0.25, "#3b0f6f");
  gradient.addColorStop(0.50, "#b63679");
  gradient.addColorStop(0.75, "#f38440");
  gradient.addColorStop(1.00, "#fcfdbf");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 1, 128);
  const url = canvas.toDataURL();
  document.getElementById("magma-bar").style.background =
    `url(${{url}}) center/100% 100% no-repeat`;
}})();

// ── Build terrain mesh ──
const terrainGeo = new THREE.BufferGeometry();
const vertCount = data.nx * data.ny;
const positions = new Float32Array(vertCount * 3);
const colors = new Float32Array(vertCount * 3);

let idx = 0;
for (let iy = 0; iy < data.ny; iy++) {{
  for (let ix = 0; ix < data.nx; ix++) {{
    const pi = iy * data.nx + ix;
    positions[idx * 3] = data.gx[ix];
    positions[idx * 3 + 1] = data.heights[pi];
    positions[idx * 3 + 2] = data.gy[iy];
    colors[idx * 3] = data.u_colors[pi * 3];
    colors[idx * 3 + 1] = data.u_colors[pi * 3 + 1];
    colors[idx * 3 + 2] = data.u_colors[pi * 3 + 2];
    idx++;
  }}
}}

const indices = [];
for (let iy = 0; iy < data.ny - 1; iy++) {{
  for (let ix = 0; ix < data.nx - 1; ix++) {{
    const a = iy * data.nx + ix;
    const b = a + 1;
    const c = a + data.nx;
    const d = c + 1;
    indices.push(a, b, d);
    indices.push(a, d, c);
  }}
}}

terrainGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
terrainGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
terrainGeo.setIndex(indices);
terrainGeo.computeVertexNormals();

const terrainMat = new THREE.MeshStandardMaterial({{
  vertexColors: true,
  roughness: 0.55,
  metalness: 0.05,
  side: THREE.DoubleSide,
}});
const terrain = new THREE.Mesh(terrainGeo, terrainMat);
terrain.receiveShadow = true;
scene.add(terrain);

// ── Wireframe overlay ──
const wireMat = new THREE.MeshBasicMaterial({{
  color: 0x0a1120,
  wireframe: true,
  transparent: true,
  opacity: 0.06,
  depthTest: true,
}});
const wireframe = new THREE.Mesh(terrainGeo, wireMat);
wireframe.position.y += 0.1;
scene.add(wireframe);

// ── Streamlines ──
data.streamlines.forEach((sl) => {{
  const pts = [];
  sl.forEach((pt) => pts.push(pt[0], pt[1], pt[2]));
  if (pts.length < 6) return;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  const mat = new THREE.LineBasicMaterial({{
    color: 0x88ccff,
    transparent: true,
    opacity: 0.45,
    depthTest: true,
    depthWrite: true,
  }});
  const line = new THREE.Line(geo, mat);
  line.renderOrder = 1;
  scene.add(line);
}});

// ── Glow streamlines (additive, thicker) ──
data.streamlines.forEach((sl) => {{
  const pts = [];
  sl.forEach((pt) => pts.push(pt[0], pt[1], pt[2]));
  if (pts.length < 6) return;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  const mat = new THREE.LineBasicMaterial({{
    color: 0x3388bb,
    transparent: true,
    opacity: 0.18,
    depthTest: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  }});
  const line = new THREE.Line(geo, mat);
  line.renderOrder = 2;
  scene.add(line);
}});

// ── Cell scatter points ──
const scatterGeo = new THREE.BufferGeometry();
const scatterPos = new Float32Array(data.points.length * 3);
const scatterCol = new Float32Array(data.points.length * 3);

const clusterPalette = [
  [1.00, 0.20, 0.45], [0.10, 0.75, 1.00], [0.35, 0.90, 0.25],
  [1.00, 0.70, 0.05], [0.65, 0.15, 1.00], [0.05, 0.90, 0.85],
  [1.00, 0.42, 0.10], [0.15, 0.78, 0.60], [0.90, 0.15, 0.55],
  [0.45, 0.55, 0.90], [0.90, 0.85, 0.15], [0.20, 0.65, 0.95],
];

data.points.forEach((p, i) => {{
  scatterPos[i * 3] = p.x;
  scatterPos[i * 3 + 1] = p.z + 0.3;
  scatterPos[i * 3 + 2] = p.y;
  const col = clusterPalette[p.c % clusterPalette.length];
  scatterCol[i * 3] = col[0];
  scatterCol[i * 3 + 1] = col[1];
  scatterCol[i * 3 + 2] = col[2];
}});

scatterGeo.setAttribute("position", new THREE.BufferAttribute(scatterPos, 3));
scatterGeo.setAttribute("color", new THREE.BufferAttribute(scatterCol, 3));

const scatterMat = new THREE.PointsMaterial({{
  size: 0.30,
  vertexColors: true,
  depthTest: true,
  depthWrite: true,
  blending: THREE.NormalBlending,
  transparent: true,
  opacity: 0.82,
}});
const scatter = new THREE.Points(scatterGeo, scatterMat);
scatter.renderOrder = 0;
scene.add(scatter);

// ── Floor reference plane ──
const floorGeo = new THREE.PlaneGeometry(100, 100);
floorGeo.rotateX(-Math.PI / 2);
const floorMat = new THREE.MeshBasicMaterial({{
  color: 0x060912,
  side: THREE.DoubleSide,
  transparent: true,
  opacity: 0.6,
  depthWrite: false,
}});
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.position.y = -0.8;
scene.add(floor);

// ── Raycaster for tooltip ──
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById("tooltip");

window.addEventListener("mousemove", (e) => {{
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(scatter);
  if (intersects.length > 0) {{
    const idx = intersects[0].index;
    const pt = data.points[idx];
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX + 16) + "px";
    tooltip.style.top = (e.clientY - 12) + "px";
    tooltip.innerHTML =
      `Cluster ${{pt.c + 1}}<br>Uncertainty ${{pt.u.toFixed(3)}}<br>Height ${{pt.z.toFixed(1)}}`;
  }} else {{
    tooltip.style.display = "none";
  }}
}});

// ── Resize ──
window.addEventListener("resize", () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ── Animate ──
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  terrain 3d  → {output_path}")
