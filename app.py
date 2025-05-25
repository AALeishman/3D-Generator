import streamlit as st
import numpy as np
import trimesh
from io import BytesIO
import random
st.set_page_config(page_title="Geometric Vase Generator", layout="centered")
st.title("Geometric Vase Generator :amphora:")
st.markdown("Create 3D-printable spiral vases and download STL files. Max size: 180×180×180 mm.")
# --- Randomize logic ---
if "randomize" not in st.session_state:
    st.session_state.randomize = False
def trigger_random():
    st.session_state.randomize = True
st.button(":game_die: Randomize Vase", on_click=trigger_random)
# --- Parameter setup (manual or random) ---
if st.session_state.randomize:
    height = random.randint(80, 180)
    base_radius = random.randint(20, 80)
    top_radius = random.randint(10, 60)
    wall_thickness = random.randint(1, 4)
    twist_freq = random.randint(3, 12)
    st.session_state.randomize = False
else:
    height = st.slider("Height (mm)", 50, 180, 120)
    base_radius = st.slider("Base Radius (mm)", 10, 90, 30)
    top_radius = st.slider("Top Radius (mm)", 5, 90, 15)
    wall_thickness = st.slider("Wall Thickness (mm)", 1, 5, 2)
    twist_freq = st.slider("Twist Frequency (waves)", 1, 20, 6)
# --- Mesh generation ---
resolution = 200
segments_height = 100
theta = np.linspace(0, 2 * np.pi, resolution)
z = np.linspace(0, height, segments_height)
theta_grid, z_grid = np.meshgrid(theta, z)
r_grid = np.linspace(base_radius, top_radius, segments_height).reshape(-1, 1)
r_outer = r_grid + 2 * np.sin(twist_freq * theta_grid + (z_grid / height) * 2 * np.pi)
x_outer = r_outer * np.cos(theta_grid)
y_outer = r_outer * np.sin(theta_grid)
z_outer = z_grid
r_inner = r_outer - wall_thickness
x_inner = r_inner * np.cos(theta_grid)
y_inner = r_inner * np.sin(theta_grid)
z_inner = z_grid
vertices = []
faces = []
def add_faces(v1, v2, v3, v4, offset):
    return [
        [v1 + offset, v2 + offset, v3 + offset],
        [v3 + offset, v4 + offset, v1 + offset]
    ]
# Outer and inner surface
for surface in [(x_outer, y_outer, z_outer), (x_inner, y_inner, z_inner)]:
    x, y, z = surface
    for i in range(segments_height - 1):
        for j in range(resolution - 1):
            v1 = len(vertices)
            vertices.append([x[i, j], y[i, j], z[i, j]])
            vertices.append([x[i+1, j], y[i+1, j], z[i+1, j]])
            vertices.append([x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]])
            vertices.append([x[i, j+1], y[i, j+1], z[i, j+1]])
            faces.extend(add_faces(0, 1, 2, 3, v1))
# Top and bottom caps
for i in range(resolution - 1):
    # Bottom
    v_center = len(vertices)
    vertices.append([0, 0, 0])
    v1 = len(vertices)
    vertices.append([x_outer[0, i], y_outer[0, i], z_outer[0, i]])
    v2 = len(vertices)
    vertices.append([x_outer[0, i+1], y_outer[0, i+1], z_outer[0, i+1]])
    faces.append([v_center, v1, v2])
    # Top
    v_center = len(vertices)
    vertices.append([0, 0, height])
    v1 = len(vertices)
    vertices.append([x_inner[-1, i+1], y_inner[-1, i+1], z_inner[-1, i+1]])
    v2 = len(vertices)
    vertices.append([x_inner[-1, i], y_inner[-1, i], z_inner[-1, i]])
    faces.append([v_center, v1, v2])
# --- Export STL ---
vase_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
max_bounds = vase_mesh.bounds[1] - vase_mesh.bounds[0]
scale_factor = min(180.0 / np.max(max_bounds), 1.0)
vase_mesh.apply_scale(scale_factor)
stl_bytes = BytesIO()
vase_mesh.export(stl_bytes, file_type='stl')
stl_bytes.seek(0)
st.download_button(":package: Download STL", stl_bytes, file_name="vase.stl", mime="model/stl")
final_size = np.round(scale_factor * max_bounds, 1)
st.info(f":bricks: Final vase size: {final_size[0]} × {final_size[1]} × {final_size[2]} mm")






