import streamlit as st
import numpy as np
import trimesh
from io import BytesIO
import random
import base64
from perlin_noise import PerlinNoise
import streamlit.components.v1 as components

# --- Page Config & Styling ---
st.set_page_config(page_title="Geometric Model Generator", page_icon=":amphora:", layout="centered")
st.markdown(
    """
    <style>
    .stApp { background-color: #000; color: #fff; }
    .css-18e3th9 { padding: 1rem 1rem 0 1rem; }
    .stButton>button, .stSelectbox>div>div>div>button, .stNumberInput>div>div>div>input {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 0.75rem 1.5rem;
        border-radius: 12px; font-size: 1rem; transition: transform 0.2s;
    }
    .stButton>button:hover, .stSelectbox>div>div>div>button:hover, .stNumberInput>div>div>div>input:hover {
        transform: scale(1.05); cursor: pointer;
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #f6d365 0%, #fda085 100%);
        color: #333; border: none; padding: 0.75rem 1.5rem;
        border-radius: 12px; font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Modern Geometric Model Generator")
st.markdown(
    "Define wall thickness and X/Y/Z dimensions, choose style options, then generate and preview your model!",
    unsafe_allow_html=True
)

# --- Inputs for thickness and dimensions ---
wall_thickness = st.number_input("Wall Thickness (mm)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
X_dim = st.number_input("X Dimension (mm)", min_value=10, max_value=200, value=100, step=1)
Y_dim = st.number_input("Y Dimension (mm)", min_value=10, max_value=200, value=100, step=1)
Z_dim = st.number_input("Z Dimension (mm)", min_value=10, max_value=200, value=150, step=1)

# --- Style & Category Selection ---
shape_options = ["round","triangle","square","pentagon","hexagon","heptagon","octagon","decagon"]
texture_options = [
    "smooth", "spiral", "swirl", "abstract", "coral", "fractal", "horizontal ribs",
    "vertical ribs", "vertical square ribs", "crosshatch ribs", "ribbed", "bubble",
    "spikey", "dimpled", "wave", "pebbled", "perforated",
    "voronoi cutout", "lattice", "gyroid"
]
category_options = ["Vase","Plant Pot","Lamp Base","Lamp Shade"]
shape_choice = st.selectbox("Shape", shape_options)
texture_choice = st.selectbox("Texture", texture_options)
category_choice = st.selectbox("Category", category_options)

if st.button("🎲 Generate & Preview Model"):
    # RNGs
    noise_gen = PerlinNoise(octaves=4, seed=random.randint(0,1000))
    rand_state = np.random.RandomState(random.randint(0,1000))

    # Geometry dimensions
    height = Z_dim
    max_base = min(X_dim, Y_dim) / 2
    base_radius = random.uniform(10, max_base * 0.9)
    top_radius = random.uniform(5, base_radius)

    # Mesh grid (periodic symmetry)
    res, seg = 200, 100
    theta    = np.linspace(0, 2*np.pi, res, endpoint=False)
    z_vals   = np.linspace(0, height, seg)
    tg, zg   = np.meshgrid(theta, z_vals)

    # Radius profile: straight walls for Plant Pots, taper otherwise
    if category_choice == "Plant Pot":
        r_prof = np.full((seg, 1), base_radius)
    else:
        r_prof = np.linspace(base_radius, top_radius, seg).reshape(-1, 1)

    # Generate support data for new textures
    # Voronoi seeds in normalized (theta/2pi, z/height) space
    voronoi_pts = rand_state.rand(30, 2)
    voronoi_thresh = 0.08
    # Lattice frequencies
    lat_hr = random.randint(4, 20)
    lat_vr = random.randint(4, 20)
    # Gyroid frequency
    gyro_f = random.uniform(1.0, 4.0)

    # Cross-section function
    def cross_sec(t):
        if shape_choice == "round":
            return np.ones_like(t)
        sides_map = {"triangle":3, "square":4, "pentagon":5,
                     "hexagon":6, "heptagon":7, "octagon":8, "decagon":10}
        return 1 + 0.15 * np.cos(sides_map[shape_choice] * t)

    # Texture pattern function
    def texture_pat(t, z_, h):
        pat = np.zeros_like(t)
        # Standard patterns
        if texture_choice == "spiral":
            freq = random.uniform(2, 8)
            pat = 2 * np.sin(freq * t + (z_/h) * 2 * np.pi)
        elif texture_choice == "swirl":
            ft, fz = random.uniform(3,6), random.uniform(1,3)
            pat = 2 * np.sin(ft * t) * np.cos(fz * (z_/h) * 2 * np.pi)
        elif texture_choice == "abstract":
            for i in range(pat.shape[0]):
                for j in range(pat.shape[1]):
                    p = noise_gen([i/pat.shape[0], j/pat.shape[1]])
                    pat[i,j] = p + np.sin(3 * t[i,j] + (z_[i,j]/h) * 3 * np.pi)
        elif texture_choice == "coral":
            for i in range(pat.shape[0]):
                for j in range(pat.shape[1]):
                    p = noise_gen([i/pat.shape[0], j/pat.shape[1]])
                    pat[i,j] = -abs(p) * 3 + p
        elif texture_choice == "fractal":
            for k in range(1,5):
                pat += (1/k) * np.sin((2**k) * t + (2**k) * (z_/h) * 2 * np.pi)
        elif texture_choice == "horizontal ribs":
            n = random.randint(4,60); pat = 3 * np.sin((z_/h) * n * np.pi)
        elif texture_choice == "vertical ribs":
            n = random.randint(4,60); pat = 2 * np.cos(n * t)
        elif texture_choice == "vertical square ribs":
            n = random.randint(4,60); pat = 3 * np.sign(np.cos(n * t))
        elif texture_choice == "crosshatch ribs":
            hr, vr = random.randint(4,60), random.randint(4,60)
            pat = 3 * np.sin((z_/h) * hr * np.pi) + 2 * np.cos(vr * t)
        elif texture_choice == "ribbed":
            pat = 1.5 * np.sin((z_/3) * np.pi)
        elif texture_choice == "bubble":
            pat = 2 * np.sin(6 * t) * np.sin(6 * (z_/h) * 2 * np.pi)
        elif texture_choice == "spikey":
            pat = 5 * np.sign(np.sin(10 * t) * np.sin(10 * (z_/h) * np.pi))
        elif texture_choice == "dimpled":
            pat = -2 * np.abs(np.sin((z_/h) * 10 * np.pi))
        elif texture_choice == "wave":
            pat = 2 * np.sin(3 * t + 2 * (z_/h) * np.pi)
        elif texture_choice == "pebbled":
            for i in range(pat.shape[0]):
                for j in range(pat.shape[1]):
                    pat[i,j] = noise_gen([i/pat.shape[0], j/pat.shape[1]])
            pat *= 1.5
        elif texture_choice == "perforated":
            if category_choice != "Plant Pot":
                mask = rand_state.rand(*t.shape) < 0.05; pat[mask] = -2
        # New patterns: Voronoi, Lattice, Gyroid
        elif texture_choice == "voronoi cutout":
            t_norm = t / (2 * np.pi)
            z_norm = z_ / h
            d = np.sqrt((t_norm[...,None] - voronoi_pts[:,0])**2 +
                        (z_norm[...,None] - voronoi_pts[:,1])**2)
            mask = np.min(d,axis=-1) < voronoi_thresh
            pat[mask] = -2
        elif texture_choice == "lattice":
            mask_h = np.abs(np.sin(lat_hr * t)) < 0.1
            mask_v = np.abs(np.sin(lat_vr * (z_/h) * np.pi)) < 0.1
            pat[np.logical_or(mask_h, mask_v)] = -2
        elif texture_choice == "gyroid":
            f = gyro_f
            pat = 1.5 * (np.sin(f * t) * np.cos(f * (z_/h) * 2 * np.pi) +
                         np.sin(f * (z_/h) * 2 * np.pi) * np.cos(f * t))
        return pat

    # Build vertices for outer and inner surfaces
    r_out = r_prof * cross_sec(tg) + texture_pat(tg, zg, height)
    x_out, y_out, z_out = r_out * np.cos(tg), r_out * np.sin(tg), zg
    r_in = r_out - wall_thickness
    x_in, y_in, z_in = r_in * np.cos(tg), r_in * np.sin(tg), zg

    # Triangulate mesh with wrapped seam
    verts, faces = [], []
    def quad(a,b,c,d,off):
        return [[a+off,b+off,c+off],[c+off,d+off,a+off]]
    for (X, Y, Z) in [(x_out, y_out, z_out), (x_in, y_in, z_in)]:
        for i in range(seg-1):
            for j in range(res):
                j2 = (j + 1) % res
                off = len(verts)
                verts.extend([
                    [X[i,   j],   Y[i,   j],   Z[i,   j]],
                    [X[i+1, j],   Y[i+1, j],   Z[i+1, j]],
                    [X[i+1, j2],  Y[i+1, j2],  Z[i+1, j2]],
                    [X[i,   j2],  Y[i,   j2],  Z[i,   j2]]
                ])
                faces.extend(quad(0,1,2,3,off))

    # Caps based on category
    cap_bottom = (category_choice != "Lamp Shade")
    cap_top    = (category_choice != "Lamp Shade")
    if cap_bottom:
        for j in range(res):
            j2 = (j + 1) % res
            c0 = len(verts)
            verts.extend([[0,0,0], [x_out[0,j],y_out[0,j],z_out[0,j]],
                          [x_out[0,j2],y_out[0,j2],z_out[0,j2]]])
            faces.append([c0,c0+1,c0+2])
    if cap_top:
        for j in range(res):
            j2 = (j + 1) % res
            c1 = len(verts)
            verts.extend([[0,0,height], [x_in[-1,j2],y_in[-1,j2],z_in[-1,j2]],
                          [x_in[-1,j],y_in[-1,j],z_in[-1,j]]])
            faces.append([c1,c1+1,c1+2])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Optimize mesh for 3D printing
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    try:
        mesh.fill_holes()
    except:
        pass

    # Scale anisotropically to target X, Y, Z
    bmin, bmax = mesh.bounds
    extents = bmax - bmin
    mesh.apply_scale([X_dim/extents[0], Y_dim/extents[1], Z_dim/extents[2]])

    # Preview with model-viewer
    glb_data = mesh.export(file_type='glb')
    b64_glb = base64.b64encode(glb_data).decode('utf-8')
    components.html(f"""
      <script type='module' src='https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js'></script>
      <model-viewer
        src='data:model/gltf-binary;base64,{b64_glb}'
        alt='3D model'
        environment-image="neutral"
        ambient-intensity="0.8"
        exposure="1.2"
        shadow-intensity="1"
        camera-controls
        auto-rotate
        auto-rotate-delay="1000"
        interaction-prompt="auto"
        style="width:100%; height:500px; background-color:#111; display:block;"
      ></model-viewer>
    """, height=520)

    # Download STL
    dims = f"{int(X_dim)}x{int(Y_dim)}x{int(Z_dim)}"
    filename = f"{category_choice}_{shape_choice}_{texture_choice}_{dims}.stl".replace(' ','_')
    buf = BytesIO(); mesh.export(buf,file_type='stl'); buf.seek(0)
    st.download_button("📦 Download STL", buf, file_name=filename, mime="model/stl")

    st.success("✔ Model generated and previewed!")
