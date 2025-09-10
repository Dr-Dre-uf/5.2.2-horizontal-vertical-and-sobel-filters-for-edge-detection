import streamlit as st
import numpy as np
from PIL import Image

# Sidebar controls
st.sidebar.header("Image Processing Parameters")
image_source = st.sidebar.radio(
    "Select Image Source:",
    ("Fluorescence (IF)", "Upload Image"),
    help="Choose between pre-loaded images or upload your own."
)

# Load images based on selection
if image_source == "Fluorescence (IF)":
    try:
        image = Image.open('assets/IFCells.jpg')
        image = np.array(image)
        print(f"Image IF loaded with size: {image.shape}")
    except FileNotFoundError as e:
        st.error(f"Image not found: {e}")
        st.stop()
else:  # Upload Image
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        print(f"Uploaded image loaded with size: {image.shape}")
    else:
        st.warning("Please upload an image.")
        st.stop()

# Define a simple 3x3 averaging filter
def apply_average_filter(img):
    kernel = np.ones((3, 3), np.float32) / 9
    filtered_img = cv2.filter2D(img, -1, kernel)  # Requires OpenCV
    return filtered_img

# Apply the filter (attempt with OpenCV)
try:
    import cv2  # Import OpenCV here to handle potential import errors
    filtered_image = apply_average_filter(image)
    st.image(filtered_image, caption="Filtered Image")

except ImportError:
    st.warning("OpenCV is not installed.  Cannot apply filter.")
    st.image(image, caption="Original Image (OpenCV not available)")  # Show original if OpenCV isn't present
except Exception as e:
    st.error(f"An error occurred during filtering: {e}")
    st.image(image, caption="Original Image (Error during filtering)")