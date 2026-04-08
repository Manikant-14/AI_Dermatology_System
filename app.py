
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from keras import backend as K

# ---- CUSTOM LOSS ----
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

st.set_page_config(page_title="AI Dermatology System", layout="wide")

st.title(" AI Dermatology Intelligence System")
st.markdown("Upload a skin lesion image and get AI-based clinical insights")

# ---- PATH ----
BASE_PATH = "/content/drive/MyDrive/Dermatology_AI_App/models/"

@st.cache_resource
def load_models():
    clf_model = tf.keras.models.load_model(
        BASE_PATH + "mobilenet_finetuned_final.keras"
    )

    seg_model = tf.keras.models.load_model(
        BASE_PATH + "unet_segmentation_best.keras",
        custom_objects={"bce_dice_loss": bce_dice_loss},
        compile=False
    )

    return clf_model, seg_model

clf_model, seg_model = load_models()

# ---- CLASS LABELS ----
class_map = {
    'melanoma': 'Melanoma (Skin Cancer)',
    'nevus': 'Melanocytic Nevus (Mole)',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis / Bowen’s Disease',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'vasc': 'Vascular Lesion'
}
class_keys = list(class_map.keys())

# ---- UPLOAD ----
uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=400)

    # ---- SEGMENTATION (128x128) ----
    img_seg = cv2.resize(image, (128, 128)) / 255.0
    img_seg_input = np.expand_dims(img_seg, axis=0)

    mask = seg_model.predict(img_seg_input)[0]
    mask = (mask > 0.5).astype(np.uint8)

    # ---- POST-PROCESSING ----
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ---- RESIZE MASK → 224 ----
    mask_resized = cv2.resize(mask, (224, 224))

    # ---- CLASSIFICATION INPUT ----
    img_clf = cv2.resize(image, (224, 224)) / 255.0

    segmented = img_clf * np.expand_dims(mask_resized, axis=-1)

    # ---- OVERLAY VISUALIZATION ----
    overlay = img_clf.copy()
    overlay[mask_resized == 1] = [1, 0, 0]  # red highlight (normalized)

    blended = cv2.addWeighted(img_clf, 0.7, overlay, 0.3, 0)

    with col2:
        st.subheader("Lesion Highlight (Overlay)")
        st.image(blended, width=400)

    # ---- SHOW MASK ----
    st.subheader("Segmentation Mask")
    st.image(mask_resized * 255, width=300)

    # ---- PREDICTION ----
    segmented_input = np.expand_dims(segmented, axis=0)
    prediction = clf_model.predict(segmented_input)

    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))

    disease_key = class_keys[class_idx]
    disease_name = class_map[disease_key]

    # ---- ADVANCED SEVERITY ----
    area = np.sum(mask)
    asymmetry = np.std(mask)

    severity_score = (area / (128*128)) * 0.5 + (asymmetry * 0.5)

    if severity_score < 0.25:
        risk = "LOW"
    elif severity_score < 0.5:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    st.markdown("---")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("🧬 Disease", disease_name)

    with col4:
        st.metric("📊 Confidence", f"{confidence*100:.2f}%")

    with col5:
        st.metric("⚠️ Risk Level", risk)

    # ---- CLINICAL INSIGHT ----
    st.markdown("### 🧾 Clinical Insight")

    if risk == "LOW":
        st.success("Lesion appears benign. Routine monitoring recommended.")
    elif risk == "MEDIUM":
        st.warning("Moderate risk detected. Dermatologist consultation advised.")
    else:
        st.error("High-risk lesion detected. Immediate medical attention recommended.")
