import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

# ---------------- BLACK BACKGROUND ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3, h4, h5, h6, p, span, label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title(" Dashboard Menu")
uploaded_files = st.sidebar.file_uploader(
    " Upload Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.title(" Shape & Contour Analyzer Dashboard")
st.write("Multiple image input supported using contour-based shape detection.")

# ---------------- PROCESSING ----------------
if uploaded_files:

    for idx, uploaded_file in enumerate(uploaded_files, start=1):

        st.markdown(f"## 🖼 Image {idx}")

        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)
        output_img = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        object_count = 0
        results = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            vertices = len(approx)

            if vertices == 3:
                shape = "Triangle"
            elif vertices == 4:
                x, y, w, h = cv2.boundingRect(approx)
                ar = w / float(h)
                shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if vertices == 3:
                shape = "Triangle"
            elif vertices == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    ar = w / float(h)
                    shape = "Square" if 0.90 <= ar <= 1.10 else "Rectangle"
            elif circularity > 0.70:
                    shape = "Circle"
            else:
                    shape = "Star"

            object_count += 1

            cv2.drawContours(output_img, [approx], -1, (0, 255, 255), 2)
            x, y = approx[0][0]
            cv2.putText(
                output_img, shape, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            results.append({
                "Object No": object_count,
                "Shape": shape,
                "Area (px²)": round(area, 2),
                "Perimeter (px)": round(perimeter, 2)
            })

        # ---------------- DISPLAY ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(img, width=300)

        with col2:
            st.subheader("Detected Shapes")
            st.image(output_img, width=300)

        st.metric("Objects Detected", object_count)

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No shapes detected.")

else:
    st.info(" Upload one or more images from the sidebar to start.")

