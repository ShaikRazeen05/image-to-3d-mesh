import os
import uuid
import torch
import streamlit as st
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# === Page Setup ===
st.set_page_config(page_title="🧠 3D Mesh Generator", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #0f1117; color: white; }
    h1, h2, h3 { color: #FFA500; }
    .stButton > button { background-color: #FF6F00; color: white; border-radius: 10px; }
    .css-1v0mbdj { color: white; }
    .css-ffhzg2 { background: #1e1e1e; padding: 1.5rem; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Image to 3D Mesh")
st.markdown("Upload an image, and we'll generate a 3D textured mesh using MiDaS and mesh construction!")

# === Folder Setup ===
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# === Load MiDaS ===
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    model.to(device).eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return model, transform, device

midas, transform, device = load_model()

# === File Upload ===
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    filename = f"{uuid.uuid4()}.jpg"
    input_path = os.path.join("uploads", filename)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(input_path, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔄 Generating 3D mesh..."):

        # Load & preprocess
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        input_tensor = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_tensor)
        depth = prediction.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (w, h))

        # Normalize & exaggerate depth
        depth = np.clip(depth, np.percentile(depth, 2), np.percentile(depth, 98))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 50

        # Mesh construction
        vertices = [(x - w/2, -(y - h/2), -depth[y, x]) for y in range(h) for x in range(w)]
        uvs = [(x / w, 1 - y / h) for y in range(h) for x in range(w)]
        faces = []
        for y in range(h - 1):
            for x in range(w - 1):
                i = y * w + x
                faces.append((i, i + 1, i + w))
                faces.append((i + 1, i + w + 1, i + w))

        # Save mesh
        output_dir = Path("outputs") / Path(filename).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        texture_path = output_dir / "texture.jpg"
        Image.fromarray(img).save(texture_path)

        mtl_path = output_dir / "mesh.mtl"
        with open(mtl_path, "w") as mtl:
            mtl.write("newmtl material0\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\n")
            mtl.write("Ks 0.0 0.0 0.0\nd 1.0\nillum 2\n")
            mtl.write(f"map_Kd {texture_path.name}\n")

        obj_path = output_dir / "mesh.obj"
        with open(obj_path, "w") as obj:
            obj.write(f"mtllib {mtl_path.name}\nusemtl material0\n")
            for v in vertices:
                obj.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for uv in uvs:
                obj.write(f"vt {uv[0]} {uv[1]}\n")
            for f in faces:
                v1, v2, v3 = f
                obj.write(f"f {v1+1}/{v1+1} {v2+1}/{v2+1} {v3+1}/{v3+1}\n")

    st.success("✅ 3D model generated successfully!")

    st.markdown("### 📦 Download your 3D model")
    st.download_button("⬇️ Download OBJ", open(obj_path, "rb"), file_name="mesh.obj")
    st.download_button("⬇️ Download Texture", open(texture_path, "rb"), file_name="texture.jpg")
    st.download_button("⬇️ Download MTL", open(mtl_path, "rb"), file_name="mesh.mtl")

