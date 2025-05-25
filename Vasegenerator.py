import streamlit as st
import numpy as np
import trimesh
from io import BytesIO
import random
import base64
from perlin_noise import PerlinNoise
import streamlit.components.v1 as components

# --- Page Config & Modern Styling ---
st.set_page_config(page_title="Geometric Vase Generator", page_icon=":amphora:", layout="centered")
st.markdown(
    """
    <style>
    .stApp { background-color: #000; color: #fff; }
    .css-18e3th9 { padding: 1rem 1rem 0 1rem; }
    .stButton>button, .stSelectbox>div>div>div>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 0.75rem 1.5rem;
        border-radius: 12px; font-size: 1rem; transition: transform 0.2s;
    }
    .stButton>button:hover, .stSelectbox>div>div>div>button:hover { transform: scale(1.05); cursor: pointer; }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #f6d365 0%, #fda085 100%);
        color: #333; border: none; padding: 0.75rem 1.5rem;
        border-radius: 12px; font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Modern Geometric Vase Generator")
st.markdown("Select your basic style options, then generate, preview & download your vase!", unsafe_allow_html=True)

# --- Style Selection ---
shape_options = [
    "round", "triangle", "square", "pentagon", "hexagon", "heptagon", "octagon", "decagon"
]
texture_options = [
    "smooth", "horizontal ribs", "vertical ribs", "crosshatch ribs", 
    "ribbed", "bubble", "spikey", "dimpled", "wave", "pebbled", "perforated"
]
shape_choice = st.selectbox("Shape", shape_options)
texture_choice = st.selectbox("Texture", texture_options)

MAX_DIM = 200  # mm limit

if st.button("🎲 Generate, Preview & Download Vase"):
    # ---- Random core parameters ----
    height = random.randint(50, MAX_DIM)
    base_radius = random.randint(10, MAX_DIM // 2)
    top_radius = random.randint(5, base_radius)
    wall_thickness = random.randint(1, 5)
    # Generators
    noise_gen = PerlinNoise(octaves=4, seed=random.randint(0, 1000))
    hole_rng = np.random.RandomState(random.randint(0, 1000))

    # ---- Grid setup ----
    resolution = 200
    segments_height = 100
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, segments_height)
    tg, zg = np.meshgrid(theta, z)
    r_profile = np.linspace(base_radius, top_radius, segments_height).reshape(-1, 1)

    # ---- Cross-section based on shape_choice ----
    def cross_section(t_grid):
        if shape_choice == "round": return np.ones_like(t_grid)
        sides_map = {"triangle":3, "square":4, "pentagon":5,
                     "hexagon":6, "heptagon":7, "octagon":8, "decagon":10}
        sides = sides_map.get(shape_choice, 3)
        return 1 + 0.15 * np.cos(sides * t_grid)

    # ---- Surface pattern based on texture_choice ----
    def surface_pattern(t_grid, z_grid, h):
        if texture_choice == "smooth":
            return np.zeros_like(t_grid)
        if texture_choice == "horizontal ribs":
            ribs = random.randint(4, 60)
            return 3 * np.sin((z_grid / h) * ribs * np.pi)
        if texture_choice == "vertical ribs":
            ribs = random.randint(4, 60)
            return 2 * np.cos(ribs * t_grid)
        if texture_choice == "crosshatch ribs":
            hr = random.randint(4,60)
            vr = random.randint(4,60)
            return 3 * np.sin((z_grid / h) * hr * np.pi) + 2 * np.cos(vr * t_grid)
        if texture_choice == "ribbed":
            return 1.5 * np.sin((z_grid / 3) * np.pi)
        if texture_choice == "bubble":
            return 2 * np.sin(6 * t_grid) * np.sin(6 * (z_grid / h) * 2 * np.pi)
        if texture_choice == "spikey":
            return 5 * np.sign(np.sin(10 * t_grid) * np.sin(10 * (z_grid / h) * np.pi))
        if texture_choice == "dimpled":
            return -2 * np.abs(np.sin((z_grid / h) * 10 * np.pi))
        if texture_choice == "wave":
            return 2 * np.sin(3 * t_grid + 2 * (z_grid / h) * np.pi)
        if texture_choice == "pebbled":
            pat = np.zeros_like(t_grid)
            for i in range(pat.shape[0]):
                for j in range(pat.shape[1]):
                    pat[i,j] = noise_gen([i / pat.shape[0], j / pat.shape[1]])
            return pat * 1.5
        # perforated: random holes indent
        pat = np.zeros_like(t_grid)
        mask = hole_rng.rand(*t_grid.shape) < 0.05
        pat[mask] = -2
        return pat

    # ---- Build profiles ----
    cs = cross_section(tg)
    pat = surface_pattern(tg, zg, height)
    r_out = r_profile * cs + pat
    x_out = r_out * np.cos(tg); y_out = r_out * np.sin(tg); z_out = zg
    r_in = r_out - wall_thickness
    x_in = r_in * np.cos(tg); y_in = r_in * np.sin(tg); z_in = zg

    # ---- Triangulation ----
    verts, faces = [], []
    def quad(v0, v1, v2, v3, off): return [[v0+off, v1+off, v2+off], [v2+off, v3+off, v0+off]]
    for (X, Y, Z) in [(x_out, y_out, z_out), (x_in, y_in, z_in)]:
        for i in range(segments_height - 1):
            for j in range(resolution - 1):
                idx = len(verts)
                verts.extend([
                    [X[i, j], Y[i, j], Z[i, j]],
                    [X[i+1, j], Y[i+1, j], Z[i+1, j]],
                    [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]],
                    [X[i, j+1], Y[i, j+1], Z[i, j+1]]
                ])
                faces.extend(quad(0, 1, 2, 3, idx))
    # ---- Caps ----
    for i in range(resolution - 1):
        c0 = len(verts)
        verts.extend([[0, 0, 0], [x_out[0, i], y_out[0, i], z_out[0, i]], [x_out[0, i+1], y_out[0, i+1], z_out[0, i+1]]])
        faces.append([c0, c0+1, c0+2])
        c1 = len(verts)
        verts.extend([[0, 0, height], [x_in[-1, i+1], y_in[-1, i+1], z_in[-1, i+1]], [x_in[-1, i], y_in[-1, i], z_in[-1, i]]])
        faces.append([c1, c1+1, c1+2])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    bounds = mesh.bounds[1] - mesh.bounds[0]
    scale = min(MAX_DIM / max(bounds), 1.0)
    mesh.apply_scale(scale)

    # ---- Preview via model-viewer ----
    glb = mesh.export(file_type='glb')
    b64 = base64.b64encode(glb).decode('utf-8')
    html = f"""
    <script type='module' src='https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js'></script>
    <model-viewer src='data:model/gltf-binary;base64,{b64}' alt='vase preview' auto-rotate camera-controls style='width:100%; height:400px; background-color:#000;'></model-viewer>
    """
    components.html(html, height=450)

    # ---- Download button (shape_texture_dimensions) ----
    final = np.round((bounds * scale), 1)
    dims = f"{int(final[0])}x{int(final[1])}x{int(final[2])}"
    filename = f"{shape_choice}_{texture_choice}_{dims}.stl".replace(" ", "_")
    buf = BytesIO(); mesh.export(buf, file_type='stl'); buf.seek(0)
    st.download_button("📦 Download STL", buf, file_name=filename, mime="model/stl")

    st.success(f"🔺 Generated vase: {final[0]}×{final[1]}×{final[2]} mm (HxWxD)")

