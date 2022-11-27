import streamlit as st
import numpy as np
from PIL import Image
import cv2

def pencilSketch(input_image):
    grey = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) # Meng convert gambar menjadi abu abu
    grey = cv2.medianBlur(grey, 5) #membuat gambar blurry
    edges = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7,7) #membuat sketch
    color = cv2.bilateralFilter(input_image, d=9, sigmaColor=200,sigmaSpace=200)
    # cartoon = cv2.bitwise_and(color, color, mask=edges)

    data = np.float32(color).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, label, center = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(input_image.shape)
    blurred = cv2.medianBlur(result, 5)
    cartoon_1 = cv2.bitwise_and(blurred, blurred, mask=edges)


    return(edges, cartoon_1)

st.title("To Cartoon App")
st.write('This web app is to help convert your photos to Animation Style')

# File uploader on sidebar

image = st.sidebar.file_uploader(
    "Upload your photo", type=['jpeg', 'jpg', 'png'])

col2, col3 = st.columns(2)
if image is None:
    st.write("You have not uploaded any image")
else:
    input_image = Image.open(image)
    final_sketch = pencilSketch(np.array(input_image))
    with col2:
        st.write("**Original photo**")
        st.image(image)
    with col3:
        st.write("**Output sketch Photo**")
        st.image(final_sketch[0])
    
    st.write("**Output Cartoon Photo**")
    st.image(final_sketch[1])

