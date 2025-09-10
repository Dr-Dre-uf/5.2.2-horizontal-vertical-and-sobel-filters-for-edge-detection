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

# Define custom horizontal edge filter
horiz_filter = np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]], dtype=np.float32)

# Define custom vertical edge filter
vert_filter = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]], dtype=np.float32)

# Apply filters
if filter_type == "Horizontal":
    filtered_image = cv2.filter2D(img, -1, horiz_filter)
elif filter_type == "Vertical":
    filtered_image = cv2.filter2D(img, -1, vert_filter)
elif filter_type == "Sobel":
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    filtered_image = np.uint8(np.absolute(sobelx + sobely))

# Convert to uint8 before displaying
filtered_image = np.uint8(filtered_image)

# Display results
st.subheader("Original Image")
st.image(img, caption=f"Original {image_source}")

st.subheader("Filtered Image")
st.image(filtered_image, caption=f"{filter_type} Filtered {image_source}")