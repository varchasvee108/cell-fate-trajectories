import torch
import numpy as np
import json
from sklearn.decomposition import PCA


def generate_3d_html(predictions_path: str, output_path: str):
    preds = torch.load(predictions_path, map_location="cpu")
    inputs = preds["inputs"]
    pseudotime = preds["pseudotime"]
    quantile_preds = preds["quantile_preds"]
    cluster_preds = preds["cluster_preds"]

    # Use the last token
    x = inputs[:, -1].numpy()
    pred_next = quantile_preds[:, -1, :, 1].numpy()
    q10 = quantile_preds[:, -1, :, 0].numpy()
    q90 = quantile_preds[:, -1, :, 2].numpy()
    pseudo = pseudotime[:, -1].numpy()
    clusters = cluster_preds[:, -1].numpy()

    # Subsample for performance
    n_points = x.shape[0]
    n_sample = min(15000, n_points)
    np.random.seed(42)
    indices = np.random.choice(n_points, size=n_sample, replace=False)

    x_sub = x[indices]
    pred_sub = pred_next[indices]
    pseudo_sub = pseudo[indices]
    clusters_sub = clusters[indices]

    # Uncertainty (difference between 90th and 10th percentiles)
    uncertainty = np.mean(q90[indices] - q10[indices], axis=1)

    # PCA to 2D for the horizontal layout
    pca = PCA(n_components=2, random_state=42)
    x_2d = pca.fit_transform(x_sub)
    pred_2d = pca.transform(pred_sub)

    # Scale coordinates to [-50, 50] range for Three.js
    max_val = np.abs(x_2d).max()
    x_2d = (x_2d / max_val) * 50
    pred_2d = (pred_2d / max_val) * 50

    # Pseudotime mapped to height. High (0) to low (1)
    # Waddington landscape flows downhill.
    pseudo_min, pseudo_max = pseudo_sub.min(), pseudo_sub.max()
    pseudo_norm = (pseudo_sub - pseudo_min) / (pseudo_max - pseudo_min + 1e-8)
    y_pos = (1 - pseudo_norm) * 40

    u_min, u_max = uncertainty.min(), uncertainty.max()
    u_norm = (uncertainty - u_min) / (u_max - u_min + 1e-8)

    points = []
    for i in range(n_sample):
        points.append(
            {
                "x": round(float(x_2d[i, 0]), 3),
                "y": round(float(y_pos[i]), 3),
                "z": round(float(x_2d[i, 1]), 3),
                "dx": round(float(pred_2d[i, 0] - x_2d[i, 0]), 3),
                "dz": round(float(pred_2d[i, 1] - x_2d[i, 1]), 3),
                "c": int(clusters_sub[i]),
                "u": round(float(u_norm[i]), 3),
            }
        )

    data_json = json.dumps({"points": points, "num_clusters": int(clusters.max()) + 1})

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waddington Landscape 3D</title>
    <style>
        body {{
            margin: 0;
            overflow: hidden;
            background-color: #020204;
            color: #fff;
            font-family: 'Inter', sans-serif;
        }}
        #canvas-container {{
            width: 100vw;
            height: 100vh;
        }}
        #labels-layer {{
            position: absolute;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            pointer-events: none;
            z-index: 10;
        }}
        .semantic-label {{
            position: absolute;
            transform: translate(-50%, -50%);
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 2px;
            text-transform: uppercase;
            text-shadow: 0 0 10px rgba(0,0,0,1.0);
            pointer-events: auto;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            padding-bottom: 4px;
            transition: color 0.3s;
            white-space: nowrap;
        }}
        .label-progenitor {{
            color: #00f0ff;
            font-size: 0.9rem;
            border-bottom: 2px solid rgba(0, 240, 255, 0.6);
            text-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
        }}
        .label-terminal {{
            color: #ff0055;
            border-bottom: 1px solid rgba(255, 0, 85, 0.5);
        }}
        .label-branch {{
            color: #ffcc00;
            border-bottom: 1px dashed rgba(255, 204, 0, 0.5);
        }}
        #ui-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 40px;
            box-sizing: border-box;
            background: radial-gradient(circle at center, transparent 0%, rgba(2,2,4,0.6) 100%);
            z-index: 5;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 200;
            letter-spacing: 4px;
            text-transform: uppercase;
            text-shadow: 0 0 30px rgba(255,255,255,0.4);
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1rem;
            color: #aaa;
            letter-spacing: 2px;
            font-weight: 300;
        }}
        .controls {{
            align-self: flex-end;
            text-align: right;
            font-size: 0.8rem;
            color: #555;
            letter-spacing: 1px;
        }}
    </style>
    <!-- Three.js + Postprocessing -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/LuminosityHighPassShader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
</head>
<body>

<div id="canvas-container"></div>
<div id="labels-layer"></div>

<div id="ui-layer">
    <div class="header">
        <h1>Waddington Flow</h1>
        <p>Neural Developmental Landscape</p>
    </div>
    <div class="controls">
        <p>LEFT CLICK: ORBIT &bull; RIGHT CLICK: PAN &bull; SCROLL: ZOOM</p>
    </div>
</div>

<script id="vertexShader" type="x-shader/x-vertex">
    uniform float time;
    attribute vec3 flowDir;
    attribute float cluster;
    attribute float uncertainty;
    attribute float vertexType;
    
    varying float vCluster;
    varying float vUncertainty;
    varying float vType;
    varying vec3 vPosition;

    void main() {{
        vCluster = cluster;
        vUncertainty = uncertainty;
        vType = vertexType;
        
        vec3 pos = position;
        
        float speed = 0.4;
        float phaseOffset = position.x * 0.15 + position.z * 0.15;
        float cycle = mod(time * speed + phaseOffset + position.y * 0.1, 2.0 * 3.14159);
        
        float flowPhase = (sin(cycle) + 1.0) * 0.5;
        
        // Uncertainty represents developmental instability/chaos
        // High uncertainty -> explosive chaos. Low uncertainty -> coherent stable flow.
        float chaos = pow(uncertainty, 2.0) * 20.0 + 0.5;
        vec3 noise = vec3(
            sin(time * 2.0 + position.x * 1.4),
            cos(time * 1.8 + position.y * 1.4),
            sin(time * 2.2 + position.z * 1.4)
        ) * chaos;

        // Core flowing motion down the landscape
        pos.x += flowDir.x * flowPhase * 18.0 + noise.x;
        pos.z += flowDir.z * flowPhase * 18.0 + noise.z;
        pos.y -= flowPhase * 5.0 - noise.y * 0.5; 

        // Tail elongation
        if (vertexType > 0.5) {{
            // High uncertainty = shorter, broken trails. Low uncertainty = long lineage streams.
            float trailLength = mix(7.0, 1.0, uncertainty) + sin(time * 2.0 + phaseOffset) * 1.5;
            pos.x -= flowDir.x * trailLength;
            pos.z -= flowDir.z * trailLength;
            pos.y += trailLength * 0.3; // Tails lift slightly
        }}

        vPosition = pos;

        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
    }}
</script>

<script id="fragmentShader" type="x-shader/x-fragment">
    varying float vCluster;
    varying float vUncertainty;
    varying float vType;
    varying vec3 vPosition;

    vec3 getClusterColor(float c) {{
        int ic = int(mod(c, 8.0));
        if (ic == 0) return vec3(0.0, 0.5, 1.0);   // Deep Blue
        if (ic == 1) return vec3(1.0, 0.1, 0.4);   // Neon Pink
        if (ic == 2) return vec3(0.4, 1.0, 0.2);   // Toxic Green
        if (ic == 3) return vec3(1.0, 0.7, 0.0);   // Amber
        if (ic == 4) return vec3(0.6, 0.1, 1.0);   // Violet
        if (ic == 5) return vec3(0.0, 0.9, 0.9);   // Cyan
        if (ic == 6) return vec3(1.0, 0.4, 0.0);   // Orange
        return vec3(0.1, 0.8, 0.6);                // Seafoam
    }}

    void main() {{
        vec3 baseColor = getClusterColor(vCluster);
        
        // Trail intensity fade
        float intensity = 1.0 - vType; 
        
        // Uncertainty adds chaotic white heat (diffuse instability)
        vec3 finalColor = mix(baseColor, vec3(0.9, 0.95, 1.0), pow(vUncertainty, 1.5) * 0.8);

        // Depth / height shading
        float heightFactor = smoothstep(-15.0, 40.0, vPosition.y);
        finalColor *= mix(0.15, 1.5, heightFactor);

        // Final composite
        gl_FragColor = vec4(finalColor * intensity, intensity * (1.0 - vUncertainty * 0.4));
    }}
</script>

<script>
    const sceneData = {data_json};

    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    
    // Atmospheric Depth
    scene.fog = new THREE.FogExp2(0x020204, 0.012);

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 45, 100);

    const renderer = new THREE.WebGLRenderer({{ antialias: false, alpha: false }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x020204);
    container.appendChild(renderer.domElement);

    // Cinematic Camera Controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.04;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.3; // Slower cinematic rotation
    controls.maxPolarAngle = Math.PI / 2 - 0.05;

    // Lighting for Terrain
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
    scene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight.position.set(20, 50, -20);
    scene.add(dirLight);

    const pointLight = new THREE.PointLight(0x0088ff, 0.5, 150);
    pointLight.position.set(0, 20, 0);
    scene.add(pointLight);

    // Extract structure for storytelling
    const numClusters = sceneData.num_clusters;
    const clusterCenters = new Array(numClusters).fill(0).map(() => ({{x:0, y:0, z:0, count:0}}));
    let topY = -1000;
    let topPos = new THREE.Vector3();

    // Create Particle Trails (LineSegments)
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(sceneData.points.length * 6);
    const flowDirs = new Float32Array(sceneData.points.length * 6);
    const clusters = new Float32Array(sceneData.points.length * 2);
    const uncertainties = new Float32Array(sceneData.points.length * 2);
    const vertexTypes = new Float32Array(sceneData.points.length * 2);

    sceneData.points.forEach((p, i) => {{
        // Structure metrics
        clusterCenters[p.c].x += p.x;
        clusterCenters[p.c].y += p.y;
        clusterCenters[p.c].z += p.z;
        clusterCenters[p.c].count++;
        if (p.y > topY) {{
            topY = p.y;
            topPos.set(p.x, p.y, p.z);
        }}

        // Head Vertex
        positions[i * 6] = p.x;
        positions[i * 6 + 1] = p.y;
        positions[i * 6 + 2] = p.z;
        // Tail Vertex
        positions[i * 6 + 3] = p.x;
        positions[i * 6 + 4] = p.y;
        positions[i * 6 + 5] = p.z;
        
        flowDirs[i * 6] = p.dx;
        flowDirs[i * 6 + 1] = 0; 
        flowDirs[i * 6 + 2] = p.dz;
        flowDirs[i * 6 + 3] = p.dx;
        flowDirs[i * 6 + 4] = 0; 
        flowDirs[i * 6 + 5] = p.dz;

        clusters[i * 2] = p.c;
        clusters[i * 2 + 1] = p.c;
        
        uncertainties[i * 2] = p.u;
        uncertainties[i * 2 + 1] = p.u;
        
        vertexTypes[i * 2] = 0.0; // Head
        vertexTypes[i * 2 + 1] = 1.0; // Tail
    }});

    clusterCenters.forEach(c => {{
        if(c.count > 0) {{
            c.x /= c.count;
            c.y /= c.count;
            c.z /= c.count;
        }}
    }});

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('flowDir', new THREE.BufferAttribute(flowDirs, 3));
    geometry.setAttribute('cluster', new THREE.BufferAttribute(clusters, 1));
    geometry.setAttribute('uncertainty', new THREE.BufferAttribute(uncertainties, 1));
    geometry.setAttribute('vertexType', new THREE.BufferAttribute(vertexTypes, 1));

    const material = new THREE.ShaderMaterial({{
        vertexShader: document.getElementById('vertexShader').textContent,
        fragmentShader: document.getElementById('fragmentShader').textContent,
        uniforms: {{
            time: {{ value: 0.0 }}
        }},
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    }});

    const particles = new THREE.LineSegments(geometry, material);
    scene.add(particles);

    // Create Progenitor Source Halo (Developmental Origin)
    const haloGeo = new THREE.RingGeometry(2, 7, 32);
    haloGeo.rotateX(-Math.PI / 2);
    const haloMat = new THREE.MeshBasicMaterial({{
        color: 0x00f0ff,
        transparent: true,
        opacity: 0.25,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    }});
    const sourceHalo = new THREE.Mesh(haloGeo, haloMat);
    sourceHalo.position.copy(topPos);
    sourceHalo.position.y -= 1.0;
    scene.add(sourceHalo);

    // Create Attractor Basins
    function getClusterColorJS(c) {{
        const colors = [0x0080ff, 0xff1a66, 0x66ff33, 0xffb300, 0x991aff, 0x00e6e6, 0xff6600, 0x1acc99];
        return colors[c % 8];
    }}

    const basinGeo = new THREE.CircleGeometry(5, 32);
    basinGeo.rotateX(-Math.PI / 2);

    const labels = [];
    const labelsLayer = document.getElementById('labels-layer');

    function createLabel(text, pos, className) {{
        const el = document.createElement('div');
        el.className = 'semantic-label ' + className;
        el.innerText = text;
        labelsLayer.appendChild(el);
        labels.push({{ element: el, position: pos }});
    }}

    // Stem Cell Pool Label
    createLabel("Stem Cell Pool", topPos.clone().add(new THREE.Vector3(0, 4, 0)), "label-progenitor");

    clusterCenters.forEach((c, idx) => {{
        if (c.count < 50) return;
        if (c.y < 12) {{
            // Terminal Sink Basin
            const mat = new THREE.MeshBasicMaterial({{
                color: getClusterColorJS(idx),
                transparent: true,
                opacity: 0.12,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            }});
            const basin = new THREE.Mesh(basinGeo, mat);
            basin.position.set(c.x, c.y - 1.5, c.z);
            scene.add(basin);
            createLabel(`Fate ${{idx + 1}}`, new THREE.Vector3(c.x, c.y + 3, c.z), "label-terminal");
        }} else if (c.y < 25 && c.y >= 12) {{
            // Branching / Transition Phase
            createLabel(`Branching ${{idx + 1}}`, new THREE.Vector3(c.x, c.y + 3, c.z), "label-branch");
        }}
    }});

    // Create Developmental Terrain Manifold
    const terrainGeo = new THREE.PlaneGeometry(130, 130, 70, 70);
    terrainGeo.rotateX(-Math.PI / 2);
    
    const posAttr = terrainGeo.attributes.position;
    const searchStep = Math.max(1, Math.floor(sceneData.points.length / 800)); // Sample subset for perf
    
    for (let i = 0; i < posAttr.count; i++) {{
        const vx = posAttr.getX(i);
        const vz = posAttr.getZ(i);
        
        let weightSum = 0;
        let ySum = 0;
        
        // Inverse distance weighting to form valleys/ridges
        for (let j = 0; j < sceneData.points.length; j += searchStep) {{
            const p = sceneData.points[j];
            const dx = p.x - vx;
            const dz = p.z - vz;
            const distSq = dx*dx + dz*dz;
            
            const w = 1.0 / (distSq + 2.0);
            weightSum += w;
            ySum += p.y * w;
        }}
        
        let finalY = -10;
        if (weightSum > 0) {{
            finalY = (ySum / weightSum) - 3.0; // Basin floor slightly below streams
        }}
        posAttr.setY(i, finalY);
    }}
    terrainGeo.computeVertexNormals();

    const terrainMat = new THREE.MeshStandardMaterial({{
        color: 0x06090e,
        roughness: 0.9,
        metalness: 0.1,
        wireframe: true,
        transparent: true,
        opacity: 0.15
    }});
    
    const terrainSolidMat = new THREE.MeshStandardMaterial({{
        color: 0x010203,
        roughness: 1.0,
        metalness: 0.0,
    }});

    const terrain = new THREE.Mesh(terrainGeo, terrainMat);
    const terrainSolid = new THREE.Mesh(terrainGeo, terrainSolidMat);
    terrainSolid.position.y -= 0.5;
    scene.add(terrain);
    scene.add(terrainSolid);

    // Post-Processing (Bloom for glowing energy effect)
    const renderScene = new THREE.RenderPass(scene, camera);
    const bloomPass = new THREE.UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight), 
        2.2,  // strength
        0.5,  // radius
        0.2   // threshold
    );
    
    const composer = new THREE.EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);

    // Resize handler
    window.addEventListener('resize', () => {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        composer.setSize(window.innerWidth, window.innerHeight);
    }});

    // Animation Loop
    const clock = new THREE.Clock();
    
    function animate() {{
        requestAnimationFrame(animate);
        
        const elapsedTime = clock.getElapsedTime();
        material.uniforms.time.value = elapsedTime;
        
        // Gentle scene breathing
        scene.position.y = Math.sin(elapsedTime * 0.5) * 0.5;
        
        controls.update();
        
        // Update Semantic Labels Position (3D to 2D Screen Space projection)
        labels.forEach(l => {{
            const pos = l.position.clone();
            // Offset slightly by scene breathing
            pos.y += scene.position.y;
            pos.project(camera);
            
            // Check if behind camera
            if (pos.z > 1.0) {{
                l.element.style.opacity = '0';
                return;
            }}
            
            const x = (pos.x * 0.5 + 0.5) * window.innerWidth;
            const y = (-(pos.y * 0.5) + 0.5) * window.innerHeight;
            l.element.style.left = `${{x}}px`;
            l.element.style.top = `${{y}}px`;
            
            // Scale and fade based on distance
            const dist = camera.position.distanceTo(l.position);
            const scale = Math.max(0.4, 1.2 - dist / 150);
            const alpha = Math.max(0.1, 1.0 - (dist - 50) / 100);
            
            l.element.style.transform = `translate(-50%, -50%) scale(${{scale}})`;
            l.element.style.opacity = alpha.toString();
        }});

        composer.render();
    }}

    animate();
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  3d html    → {output_path}")
