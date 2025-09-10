import os
os.environ['CV_MATMUL_CPU_ONLY'] = '1'
os.environ['OPENCV_OPENGL_NO_LOAD'] = '1'

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color
from PIL import Image

# --- Image Loading and Selection ---
image_paths = {
    "IF Cells": "assets/IFCells.jpg",
    "Blood Smear": "assets/BloodSmear.png"
}

selected_image_name = st.sidebar.selectbox("Select Image", list(image_paths.keys()))
selected_image_path = image_paths[selected_image_name]

uploaded_file = st.sidebar.file_uploader("Or Upload Your Own Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    selected_image = image  # Use the uploaded image
else:
    selected_image = cv2.imread(selected_image_path)
    if selected_image is None:
        st.error(f"Could not load image from {selected_image_path}. Check the file path.")
        st.stop()

# --- Sidebar for Parameters ---
with st.sidebar:
    st.header("Parameters")
    filter_size = st.slider("Filter Size", 3, 9, 3, step=2)  # Adjust filter size
    sobel_ksize = st.slider("Sobel Kernel Size", 3, 11, 3, step=2)  # Adjust Sobel kernel size

# --- Functions ---
def apply_edge_filter(image, filter_kernel):
    return cv2.filter2D(image, -1, filter_kernel)

def apply_sobel_operator(image, ksize):
    gray_image = color.rgb2gray(image)
    gray_image = (gray_image * 255).astype(np.uint8)

    # Sobel Operator
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Calculate magnitude
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return gray_image, edge_magnitude

# --- Horizontal Edge Filtering ---
st.header("Horizontal Edge Filtering")

horiz_filter = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]], dtype=np.float32)

IF_horiz = apply_edge_filter(selected_image, horiz_filter)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(selected_image)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(IF_horiz, cmap='gray')
axes[1].set_title("Horizontal Edges")
axes[1].axis("off")
plt.tight_layout()
st.pyplot(fig)

# --- Vertical Edge Filtering ---
st.header("Vertical Edge Filtering")

vert_filter = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]], dtype=np.float32)

IF_vert = apply_edge_filter(selected_image, vert_filter)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(selected_image)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(IF_vert, cmap='gray')
axes[1].set_title("Vertical Edges")
axes[1].axis("off")
plt.tight_layout()
st.pyplot(fig)

# --- Sobel Filtering ---
st.header("Sobel Filtering")

I_IF, E_IF = apply_sobel_operator(selected_image, sobel_ksize)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(I_IF, cmap='gray')
axes[0].set_title('Grayscale Image')
axes[0].axis('off')
axes[1].imshow(E_IF, cmap='gray')
axes[1].set_title('Edge Magnitude')
axes[1].axis('off')
plt.tight_layout()
st.pyplot(fig)