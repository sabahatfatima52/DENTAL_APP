import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Streamlit UI
st.title("🦷 Dental Procedure Detection App")
st.write("Upload an image to detect dental procedures using YOLO.")

# Step 1: Upload image
uploaded_file = st.file_uploader("Browse and select an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Step 2: Show the uploaded image before prediction
    image = Image.open(uploaded_file)
    st.image(image, caption="Selected Image", use_column_width=True)

    # Step 3: Convert PIL image to numpy array
    image_np = np.array(image)  # Convert to NumPy array

    # Ensure the image has 3 color channels (RGB)
    if len(image_np.shape) == 2:  # Convert grayscale to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # Convert RGBA to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # Step 4: Run YOLO detection when user clicks "Run Detection"
    if st.button("Run Detection"):
        results = model.predict(image_np, save=False)  # Ensure `save=False` to avoid issues

        # Draw bounding boxes on the image
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                class_id = int(box.cls[0])
                label = f"{model.names[class_id]} ({conf:.2f})"
                
                # Draw rectangle and label
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert OpenCV image to PIL format for display
        detected_image = Image.fromarray(image_np)
        st.image(detected_image, caption="Detected Image", use_column_width=True)

        # Step 5: Option to save the image
        if st.button("Save Detected Image"):
            save_path = "detected_image.png"
            detected_image.save(save_path)
            st.success(f"Image saved as {save_path} ✅")

# Footer
st.markdown("---")
st.write("Developed by SSSSS | Powered by DENTAL_PROCEDURE_DETECTION 🦷")
