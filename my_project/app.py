import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Plant Health Assistant",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The model must be inside the same folder as app.py
MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_baseline.keras"
)

IMG_SIZE = (128, 128)

CONFIDENCE_THRESHOLD = 0.60


CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


# ============================================================
# DISEASE KNOWLEDGE BASE
# ============================================================

RECOMMENDATIONS = {

    "Pepper__bell___Bacterial_spot": {
        "title": "Pepper Bacterial Spot",
        "crop": "Bell Pepper",
        "status": "Diseased",
        "pathogen": "Xanthomonas campestris pv. vesicatoria",
        "description": "A bacterial disease causing dark, water-soaked spots on pepper leaves and fruit.",
        "actions": [
            "Prune and safely discard heavily infected leaves and debris.",
            "Avoid overhead sprinkler irrigation.",
            "Ensure adequate plant spacing for good air circulation.",
            "Sanitize pruning tools between plants.",
            "Apply copper-based bactericides according to local agricultural guidance."
        ],
        "prevention": "Rotate crops with non-solanaceous species and use certified disease-free seeds."
    },

    "Pepper__bell___healthy": {
        "title": "Healthy Bell Pepper Plant",
        "crop": "Bell Pepper",
        "status": "Healthy",
        "pathogen": "None",
        "description": "No major symptoms associated with the diseases represented in this model were detected.",
        "actions": [
            "Maintain consistent soil moisture without waterlogging.",
            "Provide balanced plant nutrition.",
            "Monitor regularly for insects and leaf symptoms.",
            "Provide adequate sunlight."
        ],
        "prevention": "Keep beds clean and inspect new seedlings before transplanting."
    },

    "Potato___Early_blight": {
        "title": "Potato Early Blight",
        "crop": "Potato",
        "status": "Diseased",
        "pathogen": "Alternaria solani",
        "description": "A fungal disease commonly producing brown-to-black leaf spots with concentric rings.",
        "actions": [
            "Remove severely affected lower leaves.",
            "Water at the base of the plant.",
            "Use mulch to reduce soil splash.",
            "Apply approved fungicides according to label instructions.",
            "Maintain adequate plant nutrition."
        ],
        "prevention": "Practice crop rotation and remove volunteer potato and nightshade plants."
    },

    "Potato___Late_blight": {
        "title": "Potato Late Blight",
        "crop": "Potato",
        "status": "Diseased",
        "pathogen": "Phytophthora infestans",
        "description": "A rapidly developing disease producing water-soaked to dark lesions under cool and wet conditions.",
        "actions": [
            "Remove severely infected foliage.",
            "Avoid overhead watering.",
            "Avoid working with wet foliage.",
            "Use fungicides recommended by local agricultural authorities.",
            "Monitor neighboring potato and tomato plants."
        ],
        "prevention": "Use certified seed tubers and resistant varieties where available."
    },

    "Potato___healthy": {
        "title": "Healthy Potato Plant",
        "crop": "Potato",
        "status": "Healthy",
        "pathogen": "None",
        "description": "No major symptoms associated with the potato diseases represented in this model were detected.",
        "actions": [
            "Maintain appropriate soil moisture.",
            "Maintain good drainage.",
            "Monitor regularly for insects and disease symptoms.",
            "Continue appropriate hilling and crop maintenance."
        ],
        "prevention": "Use certified seed potatoes and maintain clean crop rows."
    },

    "Tomato_Bacterial_spot": {
        "title": "Tomato Bacterial Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Xanthomonas spp.",
        "description": "A bacterial disease that causes small dark spots on tomato leaves.",
        "actions": [
            "Remove severely affected lower leaves during dry conditions.",
            "Use drip or base watering instead of overhead irrigation.",
            "Sanitize gardening tools.",
            "Avoid handling plants when foliage is wet.",
            "Use approved copper-based treatments according to local guidance."
        ],
        "prevention": "Use disease-free seed and practice crop rotation."
    },

    "Tomato_Early_blight": {
        "title": "Tomato Early Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Alternaria linariae / Alternaria solani",
        "description": "A fungal disease producing dark circular spots that may develop concentric rings.",
        "actions": [
            "Remove severely infected lower leaves.",
            "Use mulch to reduce soil splash.",
            "Improve airflow around plants.",
            "Water at the base of the plant.",
            "Use approved fungicides according to label instructions."
        ],
        "prevention": "Use tolerant varieties, good spacing, sanitation, and crop rotation."
    },

    "Tomato_Late_blight": {
        "title": "Tomato Late Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Phytophthora infestans",
        "description": "An aggressive disease that can rapidly damage tomato foliage under cool and wet conditions.",
        "actions": [
            "Remove severely infected plants or foliage.",
            "Keep foliage dry where possible.",
            "Avoid overhead watering.",
            "Use locally approved fungicides when appropriate.",
            "Monitor nearby potato and tomato plants."
        ],
        "prevention": "Use resistant varieties where available and avoid infected planting material."
    },

    "Tomato_Leaf_Mold": {
        "title": "Tomato Leaf Mold",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Passalora fulva",
        "description": "A fungal disease favored by high humidity and poor airflow.",
        "actions": [
            "Improve ventilation.",
            "Reduce dense foliage.",
            "Water at the soil surface.",
            "Remove severely affected leaves.",
            "Use approved fungicides when appropriate."
        ],
        "prevention": "Maintain plant spacing and select resistant varieties where available."
    },

    "Tomato_Septoria_leaf_spot": {
        "title": "Tomato Septoria Leaf Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Septoria lycopersici",
        "description": "A fungal leaf spot disease producing numerous small circular lesions.",
        "actions": [
            "Remove affected lower leaves.",
            "Use mulch to reduce soil splash.",
            "Water at the base of plants.",
            "Improve airflow.",
            "Use approved fungicides according to local guidance."
        ],
        "prevention": "Practice crop rotation and remove infected plant debris."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "title": "Tomato Spider Mites",
        "crop": "Tomato",
        "status": "Infested",
        "pathogen": "Tetranychus urticae",
        "description": "Tiny pests that can cause yellow stippling, bronzing, and webbing on tomato foliage.",
        "actions": [
            "Inspect leaf undersides carefully.",
            "Wash foliage with water where appropriate.",
            "Use appropriate insecticidal or horticultural products according to label instructions.",
            "Monitor populations regularly.",
            "Remove severely affected foliage."
        ],
        "prevention": "Reduce plant stress and monitor regularly for mites."
    },

    "Tomato__Target_Spot": {
        "title": "Tomato Target Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Corynespora cassiicola",
        "description": "A fungal disease producing brown circular lesions that may develop concentric rings.",
        "actions": [
            "Remove severely affected lower foliage.",
            "Improve airflow.",
            "Reduce soil splash.",
            "Maintain good weed control.",
            "Use approved fungicides when appropriate."
        ],
        "prevention": "Practice crop rotation and remove old crop residue."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Tomato Yellow Leaf Curl Virus",
        "description": "A viral disease associated with leaf curling, yellowing, reduced leaf size, and plant stunting.",
        "actions": [
            "Remove severely infected plants.",
            "Monitor and manage whitefly populations.",
            "Use insect-exclusion netting for young plants where practical.",
            "Control nearby weeds that may host pests.",
            "Use resistant varieties where available."
        ],
        "prevention": "Use certified planting material and resistant varieties where available."
    },

    "Tomato__Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Virus",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Tomato mosaic virus",
        "description": "A viral disease associated with mottled light and dark green leaf patterns and distorted growth.",
        "actions": [
            "Remove severely infected plants.",
            "Wash hands after handling infected plants.",
            "Disinfect gardening tools.",
            "Avoid spreading plant sap between plants.",
            "Use resistant planting material where available."
        ],
        "prevention": "Use disease-free seeds and maintain good sanitation."
    },

    "Tomato_healthy": {
        "title": "Healthy Tomato Plant",
        "crop": "Tomato",
        "status": "Healthy",
        "pathogen": "None",
        "description": "No major symptoms associated with the tomato diseases represented in this model were detected.",
        "actions": [
            "Maintain consistent soil moisture.",
            "Provide balanced plant nutrition.",
            "Maintain good airflow.",
            "Monitor foliage regularly."
        ],
        "prevention": "Maintain sanitation, crop rotation, and regular plant inspection."
    }
}


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
<style>

.main .block-container {
    max-width: 800px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

.hero-card {
    background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    color: #f8fafc;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(16, 185, 129, 0.25);
    text-align: center;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    color: #ecfdf5;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #a7f3d0;
    margin-bottom: 0;
    line-height: 1.5;
}

.guide-box {
    background: rgba(30, 41, 59, 0.7);
    border-radius: 12px;
    border: 1px solid #334155;
    padding: 1.5rem;
    margin-top: 1rem;
}

.badge-healthy {
    display: inline-block;
    background-color: #059669;
    color: #ffffff;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}

.badge-warning {
    display: inline-block;
    background-color: #d97706;
    color: #ffffff;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}

.badge-danger {
    display: inline-block;
    background-color: #dc2626;
    color: #ffffff;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}

</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource(show_spinner="Loading AI plant disease model...")
def load_model():

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found: "
            f"{MODEL_PATH}\n\n"
            "Make sure plant_disease_baseline.keras "
            "is inside the same folder as app.py."
        )

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================================
# PREDICTION
# ============================================================

def predict_disease(image):

    start_time = time.time()

    image_resized = tf.image.resize(
        image,
        IMG_SIZE
    )

    image_array = tf.expand_dims(
        image_resized,
        axis=0
    )

    raw_predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    inference_ms = (
        time.time() - start_time
    ) * 1000

    top_indices = np.argsort(
        raw_predictions
    )[::-1]

    predicted_index = int(
        top_indices[0]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        raw_predictions[predicted_index] * 100
    )

    top3 = [
        (
            CLASS_NAMES[int(idx)],
            float(raw_predictions[int(idx)] * 100)
        )
        for idx in top_indices[:3]
    ]

    return (
        predicted_class,
        confidence,
        top3,
        inference_ms
    )


# ============================================================
# LEAF VALIDATION
# ============================================================

def is_likely_leaf_image(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    green = (
        (H >= 25)
        & (H <= 100)
        & (S >= 45)
        & (V >= 35)
    )

    brown_yellow = (
        (H >= 5)
        & (H <= 40)
        & (S >= 45)
        & (V >= 35)
    )

    leaf_like_ratio = float(
        np.mean(green | brown_yellow)
    )

    saturated_ratio = float(
        np.mean(
            (S >= 45)
            & (V >= 35)
        )
    )

    return (
        leaf_like_ratio >= 0.10
        and saturated_ratio >= 0.12
    )


# ============================================================
# SEVERITY ESTIMATION
# ============================================================

def estimate_severity(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1]

    leaf_mask = saturation > 40

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            leaf_mask,
            connectivity=8
        )
    )

    if num_labels > 1:

        largest_label = (
            1
            + np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )

        leaf_mask = (
            labels == largest_label
        ).astype(np.uint8)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    brown = (
        (H >= 5)
        & (H <= 35)
        & (S >= 45)
        & (V >= 40)
    )

    dark = (
        (V < 90)
        & (S > 30)
    )

    yellow = (
        (H >= 20)
        & (H <= 40)
        & (S >= 50)
        & (V >= 100)
    )

    affected_mask = (
        (brown | dark | yellow)
        & leaf_mask.astype(bool)
    )

    affected_mask = cv2.morphologyEx(
        affected_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    leaf_pixels = np.sum(leaf_mask)

    affected_pixels = np.sum(affected_mask)

    if leaf_pixels == 0:
        return 0.0, "Unknown"

    affected_percentage = (
        affected_pixels
        / leaf_pixels
        * 100
    )

    affected_percentage = min(
        max(
            float(affected_percentage),
            0.0
        ),
        100.0
    )

    if affected_percentage < 10:
        severity = "Low"
    elif affected_percentage < 30:
        severity = "Moderate"
    elif affected_percentage < 60:
        severity = "High"
    else:
        severity = "Severe"

    return (
        affected_percentage,
        severity
    )


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_diagnostic_report(
    disease_key,
    confidence,
    affected_pct,
    severity,
    top3,
    is_confident=True
):

    rec = RECOMMENDATIONS.get(
        disease_key,
        {}
    )

    title = rec.get(
        "title",
        disease_key.replace("_", " ")
    )

    crop = rec.get(
        "crop",
        "Unknown"
    )

    pathogen = rec.get(
        "pathogen",
        "N/A"
    )

    description = rec.get(
        "description",
        "No description available."
    )

    actions = rec.get(
        "actions",
        []
    )

    prevention = rec.get(
        "prevention",
        "No prevention guidelines."
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if is_confident:

        status_label = (
            "CONFIDENT PREDICTION "
            "(Confidence >= 60%)"
        )

        diagnosis_header = (
            f"Primary Diagnosis : {title}"
        )

        action_header = (
            "RECOMMENDED ACTIONS:"
        )

        action_body = "\n".join(
            [
                f"  [{i}] {act}"
                for i, act in enumerate(
                    actions,
                    start=1
                )
            ]
        )

        prevention_body = (
            f"  • {prevention}"
        )

    else:

        status_label = (
            "UNCERTAIN / UNCONFIRMED "
            "(Confidence < 60%)"
        )

        diagnosis_header = (
            f"Top Candidate (Unconfirmed): {title}"
        )

        action_header = (
            "LOW-CONFIDENCE GUIDANCE:"
        )

        action_body = (
            "  [!] The AI is not sufficiently "
            "confident in this prediction.\n"
            "  [!] Do not apply disease-specific "
            "treatments based on this result.\n"
            "  [!] Upload a clearer image of a "
            "Tomato, Potato, or Bell Pepper leaf."
        )

        prevention_body = (
            "  • Use a clear, focused image "
            "with good lighting.\n"
            "  • Make sure the crop is Tomato, "
            "Potato, or Bell Pepper.\n"
            "  • Consult a local agricultural "
            "professional if symptoms persist."
        )

    lines = [
        "=" * 60,
        "AI PLANT HEALTH ASSISTANT - DIAGNOSTIC REPORT",
        "=" * 60,
        f"Generated Timestamp : {timestamp}",
        "Inference Model : Plant Disease CNN (128x128)",
        f"Diagnostic Status : {status_label}",
        f"Target Crop Group : {crop}",
        diagnosis_header,
        f"Suspected Pathogen : {pathogen}",
        (
            f"Model Confidence : "
            f"{confidence:.2f}% "
            f"(Threshold: "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}%)"
        ),
        (
            f"Leaf Area Affected : "
            f"{affected_pct:.2f}% (Estimated)"
        ),
        f"Severity Rating : {severity}",
        "-" * 60,
        "TOP PREDICTION PROBABILITIES:"
    ]

    for rank, (name, probability) in enumerate(
        top3,
        start=1
    ):
        lines.append(
            f"  {rank}. "
            f"{name.replace('_', ' ')} : "
            f"{probability:.2f}%"
        )

    lines.extend([
        "-" * 60,
        "PATHOLOGY DESCRIPTION:",
        description,
        "-" * 60,
        action_header,
        action_body,
        "-" * 60,
        "PREVENTION & NEXT STEPS:",
        prevention_body,
        "=" * 60,
        "DISCLAIMER: This diagnostic estimate is generated by AI and computer vision models.",
        "For important crop decisions, consult a local agricultural extension professional.",
        "=" * 60
    ])

    return "\n".join(lines)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "### 🌿 Plant Health AI"
    )

    st.caption(
        "AI-powered leaf disease diagnosis & care."
    )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🌱 AI Plant Health Assistant</div>
        <p class="hero-subtitle">
            Upload a plant leaf image and the AI will estimate the plant condition.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help=(
        "Upload a clear photo of an individual "
        "Tomato, Potato, or Bell Pepper leaf."
    )
)


input_image_rgb = None
image_source_label = ""


if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    decoded = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if decoded is not None:

        input_image_rgb = cv2.cvtColor(
            decoded,
            cv2.COLOR_BGR2RGB
        )

        image_source_label = (
            f"Uploaded: {uploaded_file.name}"
        )


# ============================================================
# ANALYSIS
# ============================================================

if input_image_rgb is not None:

    # --------------------------------------------------------
    # LEAF INSPECTION
    # --------------------------------------------------------

    st.subheader(
        "🍃 Leaf Inspection"
    )

    st.image(
        input_image_rgb,
        caption=image_source_label,
        use_container_width=True
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not is_likely_leaf_image(
        input_image_rgb
    ):

        st.divider()

        st.subheader(
            "📊 Analysis Results"
        )

        st.warning(
            "🚫 **Unsupported image detected**\n\n"
            "The uploaded image does not appear to "
            "contain a clear plant leaf.\n\n"
            "This AI model supports **Tomato, Potato, "
            "and Bell Pepper leaves**.\n\n"
            "👉 Please upload a clear, well-lit photo "
            "of a supported plant leaf."
        )

        st.stop()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing plant leaf condition..."
    ):

        (
            predicted_disease,
            confidence,
            top3_preds,
            latency_ms
        ) = predict_disease(
            input_image_rgb
        )


    # --------------------------------------------------------
    # RESULT INFORMATION
    # --------------------------------------------------------

    rec_info = RECOMMENDATIONS.get(
        predicted_disease,
        {}
    )

    plant_crop = rec_info.get(
        "crop",
        "Solanaceous Plant"
    )

    disease_title = rec_info.get(
        "title",
        predicted_disease.replace(
            "_",
            " "
        )
    )

    health_status = rec_info.get(
        "status",
        "Unknown"
    )


    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    if health_status == "Healthy":

        affected_pct = 0.0
        severity_level = "None"

    else:

        (
            affected_pct,
            severity_level
        ) = estimate_severity(
            input_image_rgb
        )


    # --------------------------------------------------------
    # CONFIDENCE GUARD
    # --------------------------------------------------------

    is_confident = (
        confidence
        >= CONFIDENCE_THRESHOLD * 100
    )


    # --------------------------------------------------------
    # RESULTS HEADER
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📊 Analysis Results"
    )


    if is_confident:

        if health_status == "Healthy":

            badge_html = (
                '<span class="badge-healthy">'
                '🟢 HEALTHY PLANT'
                '</span>'
            )

        elif severity_level in [
            "High",
            "Severe"
        ]:

            badge_html = (
                '<span class="badge-danger">'
                '🔴 DISEASE DETECTED • '
                f'{severity_level.upper()}'
                '</span>'
            )

        else:

            badge_html = (
                '<span class="badge-warning">'
                '🟠 DISEASE DETECTED • '
                f'{severity_level.upper()}'
                '</span>'
            )


        st.markdown(
            f"### {disease_title} &nbsp; {badge_html}",
            unsafe_allow_html=True
        )

        st.caption(
            f"**Target Crop:** {plant_crop} "
            f"&nbsp;•&nbsp; "
            f"**Pathogen:** "
            f"{rec_info.get('pathogen', 'N/A')}"
        )

    else:

        badge_html = (
            '<span class="badge-warning">'
            '⚠️ UNCERTAIN DIAGNOSIS'
            '</span>'
        )

        st.markdown(
            f"### Uncertain Prediction &nbsp; {badge_html}",
            unsafe_allow_html=True
        )

        st.caption(
            "The AI confidence is below the "
            "reliability threshold (60%)."
        )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    with col1:

        if is_confident:

            st.metric(
                "Disease",
                disease_title
            )

        else:

            st.metric(
                "Top Candidate",
                f"{disease_title} (Unconfirmed)"
            )

        st.metric(
            "Estimated Affected Area",
            f"{affected_pct:.2f}%"
        )


    with col2:

        if confidence >= 85:

            confidence_label = "High Certainty"

        elif is_confident:

            confidence_label = "Moderate (>=60%)"

        else:

            confidence_label = "Low (<60%)"


        st.metric(
            "Confidence",
            f"{confidence:.2f}%",
            delta=confidence_label,
            delta_color=(
                "normal"
                if is_confident
                else "inverse"
            )
        )

        st.metric(
            "Estimated Severity",
            severity_level
        )


    # --------------------------------------------------------
    # TOP PREDICTIONS
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "##### 📈 Top Prediction Probabilities"
    )

    chart_df = pd.DataFrame({
        "Condition": [
            c[0].replace("_", " ")
            for c in top3_preds
        ],
        "Probability (%)": [
            round(c[1], 2)
            for c in top3_preds
        ]
    }).sort_values(
        "Probability (%)",
        ascending=True
    )


    st.bar_chart(
        chart_df.set_index("Condition"),
        horizontal=True,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CONFIDENCE WARNING
    # --------------------------------------------------------

    if not is_confident:

        st.warning(
            "⚠️ **Low-confidence result**\n\n"
            "The AI is not sufficiently confident "
            "in this prediction.\n\n"
            "The result is **unconfirmed** and should "
            "not be treated as a confirmed disease diagnosis.\n\n"
            "👉 Upload a clear image of a Tomato, "
            "Potato, or Bell Pepper leaf under "
            "good lighting."
        )

    elif confidence >= 90:

        st.success(
            "🎯 **High Confidence:** "
            "The model produced a high-confidence prediction."
        )


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.subheader(
        "🌱 Treatment & Recommendations"
    )


    (
        act_tab1,
        act_tab2,
        act_tab3,
        act_tab4
    ) = st.tabs([
        "💊 Recommended Actions",
        "📖 Disease Description",
        "🛡️ Prevention & Care",
        "📥 Export Report"
    ])


    # --------------------------------------------------------
    # ACTIONS TAB
    # --------------------------------------------------------

    with act_tab1:

        if is_confident:

            st.markdown(
                f"#### Recommended Actions for **{disease_title}**"
            )

            actions_list = rec_info.get(
                "actions",
                []
            )

            for idx, action in enumerate(
                actions_list,
                start=1
            ):

                st.markdown(
                    f"**{idx}.** {action}"
                )

        else:

            st.markdown(
                "#### Recommended Guidance"
            )

            st.info(
                "⚠️ **Do not apply disease-specific "
                "treatments based on this result.**\n\n"
                "The confidence is below 60%, so "
                "the prediction is unconfirmed."
            )

            st.markdown(
                """
                **Next steps:**

                - Verify the crop is Tomato, Potato, or Bell Pepper.
                - Take a clear, focused image.
                - Use good natural lighting.
                - Keep the leaf fully visible.
                - Consult a local agricultural specialist if symptoms persist.
                """
            )


    # --------------------------------------------------------
    # DESCRIPTION TAB
    # --------------------------------------------------------

    with act_tab2:

        if is_confident:

            st.markdown(
                f"#### Overview: {disease_title}"
            )

            st.write(
                rec_info.get(
                    "description",
                    "No description available."
                )
            )

            st.markdown(
                f"**Identified Crop:** {plant_crop}"
            )

            st.markdown(
                f"**Pathogen:** "
                f"*{rec_info.get('pathogen', 'N/A')}*"
            )

        else:

            st.markdown(
                "#### Candidate Pathology Overview (Unconfirmed)"
            )

            st.write(
                f"The closest matching condition "
                f"was **{disease_title}**, with "
                f"{confidence:.2f}% confidence."
            )

            st.write(
                "Because confidence is below 60%, "
                "this result is unconfirmed."
            )


    # --------------------------------------------------------
    # PREVENTION TAB
    # --------------------------------------------------------

    with act_tab3:

        st.markdown(
            "#### Prevention & Cultural Practices"
        )

        if is_confident:

            st.write(
                rec_info.get(
                    "prevention",
                    "Practice good crop sanitation "
                    "and regular monitoring."
                )
            )

        else:

            st.write(
                "General plant health practices:"
            )

        st.markdown(
            """
            > **General Best Practices:**
            >
            > - Avoid unnecessary wetting of foliage.
            > - Maintain good spacing and airflow.
            > - Sanitize gardening tools regularly.
            > - Monitor plants frequently for changes.
            """
        )


    # --------------------------------------------------------
    # EXPORT TAB
    # --------------------------------------------------------

    with act_tab4:

        st.markdown(
            "#### 📄 Export Diagnostic Summary"
        )

        st.write(
            "Download a timestamped diagnostic "
            "summary for your records."
        )

        report_text = generate_diagnostic_report(
            predicted_disease,
            confidence,
            affected_pct,
            severity_level,
            top3_preds,
            is_confident=is_confident
        )

        st.download_button(
            label="📥 Download Diagnostic Report (.txt)",
            data=report_text,
            file_name=(
                f"plant_health_report_"
                f"{predicted_disease}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            ),
            mime="text/plain"
        )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "👈 Please **upload a leaf image** above to begin."
    )

    st.markdown(
        """
        <div class="guide-box">
            <h3 style="margin-top: 0; color: #ecfdf5;">
                🚀 Getting Started
            </h3>

            <p style="color: #cbd5e1;">
                Follow these steps to analyze your plant leaf:
            </p>

            <ol style="color: #cbd5e1; line-height: 1.8;">
                <li>
                    <strong>Upload a Leaf Image:</strong>
                    Provide a clear JPG, JPEG, or PNG photo
                    of a single Tomato, Potato, or Pepper leaf.
                </li>

                <li>
                    <strong>AI Diagnosis:</strong>
                    The CNN classifies the image and provides
                    a confidence score.
                </li>

                <li>
                    <strong>Review Results:</strong>
                    Review the predicted condition,
                    severity estimate, recommendations,
                    and top predictions.
                </li>

                <li>
                    <strong>Export Report:</strong>
                    Download the diagnostic summary.
                </li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Plant Health Assistant • CNN + Computer Vision"
)
