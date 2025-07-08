# 🧠 Image to 3D Mesh Generator

Transform any 2D surface image into a **3D mesh model** using AI-powered monocular depth estimation.  
This project uses the **MiDaS** model to generate a depth map, then converts it into a colored 3D mesh or point cloud — all with a simple web interface powered by **Streamlit**.

---

## 🚀 Features

- 📸 Upload a single image and get a 3D representation
- 🧠 Uses MiDaS depth estimation (DPT-Hybrid)
- 🌐 Built with **PyTorch**, **OpenCV**, and **Open3D**
- 🎨 Generates **colored point clouds** and **3D meshes**
- 💻 Deployed using Streamlit for ease of use

---


## 🧪 Sample Input → Output

| Input Image | Depth Map | 3D Mesh |
|-------------|-----------|---------|
| ![input](examples/input.jpg) | ![depth](examples/depth.jpg) | ![mesh](examples/mesh.jpg) |

---

## 🔧 Technologies Used

- 🖼️ **MiDaS**: Monocular Depth Estimation by Intel ISL
- 🔬 **PyTorch**: Inference & model loading
- 📦 **OpenCV**: Image preprocessing
- 📐 **Open3D**: Point cloud and mesh generation
- 🌐 **Streamlit**: Web app frontend

---


