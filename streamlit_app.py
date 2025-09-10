import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Sidebar
st.sidebar.header("Image Processing")

# Image selection
image_source = st.sidebar.radio(
    "Select Image:",
    ("Fluorescence (IF)", "Brightfield (BF)")
)

# Load images based on selection
try:
    if image_source == "Fluorescence (IF)":
        img = Image.open('assets/IFCells.jpg')
        img = np.array(img)
    else:  # Brightfield (BF)
        img = Image.open('assets/BloodSmear.png')
        img = np.array(img)
except FileNotFoundError as e:
    st.error(f"Image not found: {e}")
    st.stop()

# Filter selection
filter_type = st.sidebar.radio(
    "Select Filter:",
    ("Horizontal", "Vertical", "Sobel")
)

# Filter size slider
filter_size = st.sidebar.slider(
    "Filter Size:",
    min_value=3,
    max_value=11,
    value=3,
    step=2
)

# Define custom horizontal edge filter
kernel_size = filter_size
horiz_filter = np.zeros((kernel_size, kernel_size), dtype=np.float32)
horiz_filter[kernel_size // 2, :] = np.ones(kernel_size)  # Dynamic coefficient creation
horiz_filter[kernel_size // 2 - 1, :] = 0
horiz_filter[kernel_size // 2 + 1, :] = -np.ones(kernel_size)  # Dynamic coefficient creation

# Define custom vertical edge filter
vert_filter = np.zeros((kernel_size, kernel_size), dtype=np.float32)
vert_filter[:, kernel_size // 2] = np.ones(kernel_size)  # Dynamic coefficient creation
vert_filter[:, kernel_size // 2 - 1] = 0
vert_filter[:, kernel_size // 2 + 1] = -np.ones(kernel_size) # Dynamic coefficient creation

# Apply filters
if filter_type == "Horizontal":
    filtered_image = cv2.filter2D(img, -1, horiz_filter)
elif filter_type == "Vertical":
    filtered_image = cv2.filter2D(img, -1, vert_filter)
elif filter_type == "Sobel":
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=filter_size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=filter_size)
    filtered_image = np.uint8(np.absolute(sobelx + sobely))

# Convert to uint8 before displaying
filtered_image = np.uint8(filtered_image)

# Display results
st.subheader("Original Image")
st.image(img, caption=f"Original {image_source}")

st.subheader("Filtered Image")
st.image(filtered_image, caption=f"{filter_type} Filtered {image_source} (Size: {filter_size})")