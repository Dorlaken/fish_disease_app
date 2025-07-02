import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Load your YOLOv8 model once at startup
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = YOLO("best.pt")

# 2. Disease info mapping
DISEASE_INFO = {
    "Bacterial Diseases": "Treat with antibiotic A; maintain water quality.",
    "Fungal Diseases":    "Use antifungal medication B; quarantine.",
    "Healthy Fish":       "No action needed; ensure stable conditions.",
    "Parasitic Diseases": "Apply anti-parasitic treatment C; clean tank.",
    "White Tail Diseases": "Trim damaged fin areas; apply medication D."
}

# 3. App title and uploader
st.title("🐟 Aquarium Fish Disease Detector")
uploaded = st.file_uploader("Upload an image of your fish", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    # 4. Run YOLOv8 inference
    results = model(img)

    # 5. Display annotated image
    annotated = results[0].plot()
    st.image(annotated, caption="Detected Diseases", use_container_width=True)

    # 6. Show detections and info
    st.markdown("### Detected Conditions")
    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = box.conf.cpu().item()
        st.write(f"**{cls_name}** — Confidence: {conf:.2f}")
        st.write(DISEASE_INFO.get(cls_name, "No additional info available."))
        st.write("---")
