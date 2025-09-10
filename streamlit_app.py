import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color

# Load Images from the 'assets' folder
try:
    IF = cv2.imread("assets/IFCells.jpg")  # Replace "IF.png" with your image file
    BF = cv2.imread("assets/BloodSmear.png")  # Replace "BF.png" with your image file
    if IF is None or BF is None:
        raise FileNotFoundError("Could not load images.  Make sure the files exist in the 'assets' folder.")
except FileNotFoundError as e:
    st.error(f"Error loading images: {e}")
    st.stop()  # Halt execution if images can't be loaded

st.title("Edge Detection in Medical Images")

st.markdown("""
*   Apply **horizontal and vertical edge filters** to detect directional features.
*   Use the **Sobel operator** to enhance edges and structural details.
*   Compare the results of different edge detection techniques.

By analyzing edge features, we can enhance image contrast and extract critical information for further medical image analysis.
""")

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

IF_horiz = apply_edge_filter(IF, horiz_filter)
BF_horiz = apply_edge_filter(BF, horiz_filter)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].imshow(IF)
axes[0, 0].set_title("Original Fluorescence Image")
axes[0, 0].axis("off")
axes[0, 1].imshow(IF_horiz, cmap='gray')
axes[0, 1].set_title("Horizontal Edges")
axes[0, 1].axis("off")
axes[1, 0].imshow(BF)
axes[1, 0].set_title("Original Brightfield Image")
axes[1, 0].axis("off")
axes[1, 1].imshow(BF_horiz, cmap='gray')
axes[1, 1].set_title("Horizontal Edges")
axes[1, 1].axis("off")
plt.tight_layout()
st.pyplot(fig)

# --- Vertical Edge Filtering ---
st.header("Vertical Edge Filtering")

vert_filter = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]], dtype=np.float32)

IF_vert = apply_edge_filter(IF, vert_filter)
BF_vert = apply_edge_filter(BF, vert_filter)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].imshow(IF)
axes[0, 0].set_title("Original Fluorescence Image")
axes[0, 0].axis("off")
axes[0, 1].imshow(IF_vert, cmap='gray')
axes[0, 1].set_title("Vertical Edges")
axes[0, 1].axis("off")
axes[1, 0].imshow(BF)
axes[1, 0].set_title("Original Brightfield Image")
axes[1, 0].axis("off")
axes[1, 1].imshow(BF_vert, cmap='gray')
axes[1, 1].set_title("Vertical Edges")
axes[1, 1].axis("off")
plt.tight_layout()
st.pyplot(fig)

# --- Sobel Filtering ---
st.header("Sobel Filtering")

I_IF, E_IF = apply_sobel_operator(IF, sobel_ksize)
I_BF, E_BF = apply_sobel_operator(BF, sobel_ksize)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes[0, 0].imshow(I_IF, cmap='gray')
axes[0, 0].set_title('Grayscale IF Image')
axes[0, 0].axis('off')
axes[0, 1].imshow(E_IF, cmap='gray')
axes[0, 1].set_title('Edge Magnitude IF')
axes[0, 1].axis('off')
axes[1, 0].imshow(I_BF, cmap='gray')
axes[1, 0].set_title('Grayscale BF Image')
axes[1, 0].axis('off')
axes[1, 1].imshow(E_BF, cmap='gray')
axes[1, 1].set_title('Edge Magnitude BF')
axes[1, 1].axis('off')
plt.tight_layout()
st.pyplot(fig)