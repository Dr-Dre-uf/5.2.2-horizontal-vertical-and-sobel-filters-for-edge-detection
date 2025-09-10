import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="Edge Detection App", layout="wide")

st.title("Edge Detection with Horizontal, Vertical & Sobel Filters")

# Warning message
st.warning("⚠️ Do not upload sensitive or personal data. Images are processed locally in this demo app.")

# Sidebar: image selection or upload
st.sidebar.header("Image Selection")
use_uploaded = st.sidebar.checkbox("Upload your own image")

uploaded_file = None
if use_uploaded:
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
else:
    image_choice = st.sidebar.selectbox(
        "Select Default Image",
        ["Fluorescence (IFCells)", "Brightfield (BloodSmear)"],
        help="Choose a sample image if you don't want to upload your own."
    )

# Sidebar: filter strength sliders
st.sidebar.header("Filter Settings")

horiz_strength = st.sidebar.slider(
    "Horizontal Filter Strength", 0.5, 5.0, 1.0, step=0.1,
    help="Adjusts the strength of the horizontal edge filter."
)

vert_strength = st.sidebar.slider(
    "Vertical Filter Strength", 0.5, 5.0, 1.0, step=0.1,
    help="Adjusts the strength of the vertical edge filter."
)

sobel_strength = st.sidebar.slider(
    "Sobel Filter Strength", 0.5, 5.0, 1.0, step=0.1,
    help="Adjusts the strength of the Sobel edge detector."
)

# Define base filters
base_horiz_filter = np.array([[1, 1, 1],
                              [0, 0, 0],
                              [-1, -1, -1]], dtype=np.float32)

base_vert_filter = np.array([[1, 0, -1],
                             [1, 0, -1],
                             [1, 0, -1]], dtype=np.float32)

# Scale filters
horiz_filter = horiz_strength * base_horiz_filter
vert_filter = vert_strength * base_vert_filter

# Image paths for defaults
bf_path = "assets/BloodSmear.png"
if_path = "assets/IFCells.jpg"

# Load image
if use_uploaded and uploaded_file is not None:
    img = np.array(Image.open(uploaded_file).convert("RGB"))
elif not use_uploaded:
    if image_choice == "Fluorescence (IFCells)":
        img = np.array(Image.open(if_path).convert("RGB"))
    else:
        img = np.array(Image.open(bf_path).convert("RGB"))
else:
    img = None

if img is not None:
    # Function to apply filter channel-wise
    def apply_filter_rgb(img, kernel):
        channels = cv2.split(img)
        filtered_channels = [cv2.filter2D(c, -1, kernel) for c in channels]
        return cv2.merge(filtered_channels)

    # Apply custom filters
    img_horiz = apply_filter_rgb(img, horiz_filter)
    img_vert = apply_filter_rgb(img, vert_filter)

    # Edge magnitude (custom filters)
    E_custom = np.sqrt(img_horiz.astype(np.float32) ** 2 + img_vert.astype(np.float32) ** 2)
    E_custom = cv2.normalize(E_custom, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Sobel edges
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel = cv2.convertScaleAbs(sobel * sobel_strength)

    # Display results
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.image(img, caption="Original", use_container_width=True)
    col2.image(img_horiz, caption="Horizontal Edges", use_container_width=True)
    col3.image(img_vert, caption="Vertical Edges", use_container_width=True)
    col4.image(E_custom, caption="Edge Magnitude", use_container_width=True)
    col5.image(sobel, caption="Sobel Edges", use_container_width=True)
else:
    st.info("Please select or upload an image to begin.")
