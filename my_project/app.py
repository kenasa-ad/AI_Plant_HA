import os
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st


# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_cnn.keras")
DATASET_PATH = r"C:\Users\HP\Documents\egate_aiml\plantvillage\PlantVillage\PlantVillage"
IMG_SIZE = (128, 128)

# Confidence threshold for reliable diagnosis (60%)
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
# DISEASE KNOWLEDGE BASE & RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {
    "Pepper__bell___Bacterial_spot": {
        "title": "Pepper Bacterial Spot",
        "crop": "Bell Pepper",
        "status": "Diseased",
        "pathogen": "Xanthomonas campestris pv. vesicatoria",
        "description": "A destructive bacterial disease causing dark, water-soaked spots on pepper leaves and fruit, leading to premature leaf drop and sunscald.",
        "actions": [
            "Prune and safely discard heavily infected lower leaves and debris.",
            "Avoid overhead sprinkler irrigation to reduce leaf wetness duration.",
            "Ensure adequate plant spacing to maximize air circulation throughout the canopy.",
            "Sanitize pruning shears, stakes, and hands between handling plants.",
            "Apply copper-based bactericides in early infection stages per local agricultural guidelines."
        ],
        "prevention": "Rotate crops with non-solanaceous species for at least 2 years and plant certified disease-free seeds."
    },
    "Pepper__bell___healthy": {
        "title": "Healthy Bell Pepper Plant",
        "crop": "Bell Pepper",
        "status": "Healthy",
        "pathogen": "None",
        "description": "No symptoms of bacterial spot or other major leaf pathology were detected. The foliage shows normal vigor and coloration.",
        "actions": [
            "Maintain consistent soil moisture without waterlogging the root zone.",
            "Provide balanced N-P-K nutrition with supplemental calcium and magnesium.",
            "Scout weekly for early signs of aphids, mites, or foliar spotting.",
            "Ensure at least 6 to 8 hours of direct daily sunlight."
        ],
        "prevention": "Keep garden beds weed-free and inspect new seedlings thoroughly before transplanting."
    },
    "Potato___Early_blight": {
        "title": "Potato Early Blight",
        "crop": "Potato",
        "status": "Diseased",
        "pathogen": "Alternaria solani",
        "description": "A common fungal disease characterized by brown-to-black necrotic spots with distinctive concentric rings ('target-board' appearance) on older leaves.",
        "actions": [
            "Remove severely blighted lower leaves to prevent spore dissemination upward.",
            "Apply drip irrigation or water at the base of stems early in the morning.",
            "Mulch around potato hills to prevent soil-borne spores from splashing onto foliage.",
            "Apply approved protectant fungicides (such as chlorothalonil or mancozeb) according to label instructions.",
            "Ensure adequate nitrogen and potassium fertilization to reduce plant stress."
        ],
        "prevention": "Practice 3-year crop rotation and destroy volunteer potato plants and nightshade weeds."
    },
    "Potato___Late_blight": {
        "title": "Potato Late Blight",
        "crop": "Potato",
        "status": "Diseased",
        "pathogen": "Phytophthora infestans",
        "description": "A severe, fast-moving water mold pathogen causing pale-to-dark water-soaked lesions that turn necrotic, often with white fungal growth on leaf undersides in humid conditions.",
        "actions": [
            "Urgently remove and bag infected foliage; do not compost diseased plants.",
            "Immediately eliminate overhead watering and avoid working in wet foliage.",
            "Apply targeted, systemic fungicides recommended by regional agricultural extension services.",
            "Hill up potatoes generously to shield developing tubers from washed-down spores.",
            "Inspect neighboring potato and tomato crops daily for rapid spot spread."
        ],
        "prevention": "Plant certified disease-free seed tubers and select resistant potato cultivars when available."
    },
    "Potato___healthy": {
        "title": "Healthy Potato Plant",
        "crop": "Potato",
        "status": "Healthy",
        "pathogen": "None",
        "description": "Foliage exhibits vibrant green color, strong structural integrity, and no detectable fungal or bacterial blight lesions.",
        "actions": [
            "Continue regular hilling and soil maintenance around potato tubers.",
            "Maintain steady moisture during tuber initiation and bulking stages.",
            "Monitor weekly for Colorado potato beetles, aphids, and flea beetles.",
            "Provide good drainage to avoid tuber rot."
        ],
        "prevention": "Use certified seed potatoes and maintain clean crop rows with adequate ventilation."
    },
    "Tomato_Bacterial_spot": {
        "title": "Tomato Bacterial Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Xanthomonas spp.",
        "description": "Bacterial pathogen that causes small (less than 3mm), dark brown-to-black angular spots with translucent water-soaked margins on tomato leaves.",
        "actions": [
            "Prune off lower infected branches during dry weather conditions.",
            "Use drip irrigation lines or soaker hoses; avoid overhead misting.",
            "Disinfect stakes, trellises, and gardening tools with 10% bleach solution.",
            "Avoid cultivating or harvesting when leaves are wet with dew or rain.",
            "Apply fixed copper sprays combined with mancozeb where approved for bacterial control."
        ],
        "prevention": "Sow certified hot-water treated seeds and practice strict 2 to 3-year non-solanaceous crop rotations."
    },
    "Tomato_Early_blight": {
        "title": "Tomato Early Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Alternaria linariae / Alternaria solani",
        "description": "A prevalent fungal issue causing dark brown circular spots with characteristic concentric rings surrounded by yellow chlorotic halos.",
        "actions": [
            "Remove lower infected leaves up to 12-18 inches above the soil line.",
            "Apply a 2-3 inch organic mulch layer beneath plants to block soil splash.",
            "Stake or cage tomato vines to keep foliage elevated and well-aerated.",
            "Apply copper or bio-fungicide sprays at early onset according to label guidance.",
            "Remove and burn or discard all crop residue at the end of the season."
        ],
        "prevention": "Select early-blight tolerant hybrids, maintain optimal plant spacing, and rotate planting beds."
    },
    "Tomato_Late_blight": {
        "title": "Tomato Late Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Phytophthora infestans",
        "description": "An aggressive, fast-spreading oomycete causing irregular olive-green to greasy brownish lesions that rapidly kill leaves and stems in cool, moist weather.",
        "actions": [
            "Isolate and immediately rogue severely infected plants to protect healthy neighbors.",
            "Keep tomato foliage completely dry; shelter plants if feasible during rainy spells.",
            "Apply approved protective or translaminar fungicides without delay.",
            "Prune non-infected foliage to maximize direct sun exposure and rapid drying.",
            "Avoid placing tomato plants close to potato patches."
        ],
        "prevention": "Grow resistant tomato varieties and avoid importing unverified transplants."
    },
    "Tomato_Leaf_Mold": {
        "title": "Tomato Leaf Mold",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Passalora fulva (Cladosporium fulvum)",
        "description": "A fungal disease thriving in high humidity (>85%) producing pale yellow spots on upper leaf surfaces and velvety olive-green/brown molds underneath.",
        "actions": [
            "Increase greenhouse/grow-room ventilation and run circulation fans.",
            "Prune excess suckers and lower dense canopy to lower localized relative humidity.",
            "Water early in the day directly into root soil; avoid wetting upper leaf surfaces.",
            "Remove affected leaves carefully in plastic bags to avoid dispersing spores.",
            "Apply preventative sulfur or copper fungicides if humidity cannot be reduced."
        ],
        "prevention": "Ensure wide plant spacing and utilize leaf-mold resistant tomato hybrids in protected cultivation."
    },
    "Tomato_Septoria_leaf_spot": {
        "title": "Tomato Septoria Leaf Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Septoria lycopersici",
        "description": "A destructive fungal leaf spot producing numerous small (2-5mm) circular lesions with dark brown margins and sunken grayish-tan centers containing tiny black specks.",
        "actions": [
            "Prune diseased lower leaves promptly upon first observation.",
            "Apply clean straw, woodchips, or plastic mulch around the root zone.",
            "Water exclusively at the base of the plant using drip or soaker methods.",
            "Spray with approved copper or chlorothalonil fungicides following rain events.",
            "Thoroughly clean up and dispose of infected plant debris in autumn."
        ],
        "prevention": "Implement a 3-year solanaceous crop rotation and sanitize all plant supports."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "title": "Tomato Spider Mites (Two-Spotted)",
        "crop": "Tomato",
        "status": "Infested",
        "pathogen": "Tetranychus urticae",
        "description": "Microscopic arachnid pests that pierce plant cells, causing fine yellow stippling, bronze speckling, and delicate silken webbing under leaf surfaces.",
        "actions": [
            "Isolate infested plants and wash foliage undersides with a strong stream of water.",
            "Apply horticultural oils, neem oil, or insecticidal soaps directly to leaf undersides.",
            "Introduce beneficial biological predators such as Phytoseiulus persimilis predatory mites.",
            "Avoid broad-spectrum synthetic pyrethroids that destroy beneficial predatory insects.",
            "Prune and discard heavily webbed terminal shoots."
        ],
        "prevention": "Avoid hot, dry, dusty conditions around plants; maintain adequate soil moisture to reduce drought stress."
    },
    "Tomato__Target_Spot": {
        "title": "Tomato Target Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Corynespora cassiicola",
        "description": "A fungal disease characterized by pinpoint brown lesions that expand into circular brown spots with concentric rings and distinct yellow halos.",
        "actions": [
            "Prune lower canopy leaves to reduce humidity and soil splash.",
            "Maintain strict weed control, especially around wild nightshade relatives.",
            "Improve trellis support to elevate all vines from the ground.",
            "Apply labeled broad-spectrum protective fungicides during humid weather cycles."
        ],
        "prevention": "Avoid continuous cropping of tomatoes or peppers in the same ground and remove old crop residue promptly."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Tomato Yellow Leaf Curl Begomovirus (Vectored by Whiteflies)",
        "description": "A debilitating viral infection causing pronounced upward leaf curling, chlorotic leaf margins, reduced leaf size (cupping), and severe plant stunting.",
        "actions": [
            "Uproot and securely bag infected plants immediately; viral infections cannot be cured once established.",
            "Deploy yellow sticky traps to detect and monitor sweetpotato whitefly (Bemisia tabaci) populations.",
            "Apply insecticidal soap, neem oil, or selective systemic insecticides to manage whitefly vectors.",
            "Cover young nursery plants with fine insect-exclusion netting (50+ mesh).",
            "Eliminate broadleaf weeds nearby that serve as secondary viral reservoirs."
        ],
        "prevention": "Plant certified TYLCV-resistant hybrids and use reflective silver mulches to repel whiteflies."
    },
    "Tomato__Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Virus (ToMV)",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Tomato Mosaic Tobamovirus",
        "description": "A highly stable, mechanically transmitted virus causing mottled light/dark green mosaic patterns, distorted blistered leaves, and 'shoestring' leaf deformation.",
        "actions": [
            "Immediately rogue and discard infected plants; do not compost them.",
            "Wash hands thoroughly with soap and water after handling tobacco or infected plants.",
            "Disinfect pruning tools in a 20% non-fat dry milk solution or trisodium phosphate (TSP).",
            "Do not smoke or use tobacco products near tomato greenhouses or gardens.",
            "Handle plants minimally during humid morning hours."
        ],
        "prevention": "Purchase ToMV-resistant seeds and avoid planting where infected crops grew the preceding season."
    },
    "Tomato_healthy": {
        "title": "Healthy Tomato Plant",
        "crop": "Tomato",
        "status": "Healthy",
        "pathogen": "None",
        "description": "The leaf shows optimal chlorophyll density, uniform green pigmentation, sturdy venation, and no signs of bacterial, fungal, viral, or mite infestation.",
        "actions": [
            "Maintain consistent, deep watering to prevent blossom end rot and growth cracks.",
            "Feed regularly with a balanced tomato fertilizer rich in potassium and micronutrients.",
            "Continue regular sucker pruning and maintain sturdy stake or cage supports.",
            "Inspect foliage weekly to spot any emergent pest or disease early."
        ],
        "prevention": "Keep ground mulched, rotate solanaceous beds every 2-3 years, and sterilize garden tools regularly."
    }
}


# ============================================================
# PAGE CONFIGURATION & DARK DASHBOARD STYLING
# ============================================================

st.set_page_config(
    page_title="AI Plant Health Assistant",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Container */
    .main .block-container {
        max-width: 800px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    
    /* Header Hero Banner */
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

    /* Card Containers */
    .dashboard-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.4rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        border: 1px solid #334155;
        margin-bottom: 1.2rem;
    }

    /* Diagnosis Badges */
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
    
    /* Getting Started Box */
    .guide-box {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        border: 1px solid #334155;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# MODEL & CLASS LOADER FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner="Loading AI plant disease model...")
def load_model():
    """Loads and caches the primary Keras plant disease CNN model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


def get_class_names():
    """Returns the alphabetically sorted list of 15 plant disease classes."""
    if CLASS_NAMES:
        return CLASS_NAMES

    if os.path.exists(DATASET_PATH):
        classes = [
            name for name in os.listdir(DATASET_PATH)
            if os.path.isdir(os.path.join(DATASET_PATH, name))
        ]
        return sorted(classes)

    return CLASS_NAMES


class_names = get_class_names()


# ============================================================
# AI INFERENCE PIPELINE
# ============================================================

def predict_disease(image):
    """
    Performs CNN model inference on the input image.
    Returns:
      - predicted_class: string label
      - confidence: float percentage of top prediction (0 to 100)
      - top3: list of tuples (class_name, confidence_percentage)
      - inference_ms: inference latency in milliseconds
    """
    start_time = time.time()

    image_resized = tf.image.resize(image, IMG_SIZE)
    image_array = tf.expand_dims(image_resized, axis=0)

    raw_predictions = model.predict(image_array, verbose=0)[0]
    inference_ms = (time.time() - start_time) * 1000

    top_indices = np.argsort(raw_predictions)[::-1]

    predicted_index = top_indices[0]
    predicted_class = class_names[predicted_index]
    confidence = float(raw_predictions[predicted_index] * 100)

    top3 = [
        (class_names[idx], float(raw_predictions[idx] * 100))
        for idx in top_indices[:3]
    ]

    return predicted_class, confidence, top3, inference_ms


# ============================================================
# OUT-OF-SCOPE / LEAF IMAGE VALIDATION
# ============================================================

def is_likely_leaf_image(image):
    """
    Lightweight pre-check to reject clearly non-plant images before
    sending them to the disease classifier.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Green vegetation and common yellow/brown leaf tissue.
    green = (H >= 25) & (H <= 100) & (S >= 45) & (V >= 35)
    brown_yellow = (H >= 5) & (H <= 40) & (S >= 45) & (V >= 35)

    leaf_like_ratio = float(np.mean(green | brown_yellow))
    saturated_ratio = float(np.mean((S >= 45) & (V >= 35)))

    # Conservative thresholds so damaged/yellow leaves can still pass.
    return leaf_like_ratio >= 0.10 and saturated_ratio >= 0.12


# ============================================================
# COMPUTER VISION LEAF SEVERITY ESTIMATION
# ============================================================

def estimate_severity(image):
    """
    Estimates the percentage of affected leaf area using HSV color filtering.
    Returns:
      - affected_percentage: float (0.0 to 100.0)
      - severity: string ("Low", "Moderate", "High", "Severe")
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    # Leaf blade segmentation
    leaf_mask = saturation > 40
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    # Retain largest connected region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_mask = (labels == largest_label).astype(np.uint8)

    # Discoloration symptom detection (Brown, Dark/Necrotic, Yellow/Chlorotic)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    brown = (H >= 5) & (H <= 35) & (S >= 45) & (V >= 40)
    dark = (V < 90) & (S > 30)
    yellow = (H >= 20) & (H <= 40) & (S >= 50) & (V >= 100)

    affected_mask = (brown | dark | yellow) & leaf_mask.astype(bool)
    affected_mask = cv2.morphologyEx(
        affected_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    leaf_pixels = np.sum(leaf_mask)
    affected_pixels = np.sum(affected_mask)

    if leaf_pixels == 0:
        return 0.0, "Unknown"

    affected_percentage = float((affected_pixels / leaf_pixels) * 100)
    affected_percentage = min(max(affected_percentage, 0.0), 100.0)

    if affected_percentage < 10:
        severity = "Low"
    elif affected_percentage < 30:
        severity = "Moderate"
    elif affected_percentage < 60:
        severity = "High"
    else:
        severity = "Severe"

    return affected_percentage, severity


# ============================================================
# DIAGNOSTIC REPORT GENERATOR
# ============================================================

def generate_diagnostic_report(disease_key, confidence, affected_pct, severity, top3, is_confident=True):
    """Formats a downloadable agronomy diagnostic summary."""
    rec = RECOMMENDATIONS.get(disease_key, {})
    title = rec.get("title", disease_key.replace("_", " "))
    crop = rec.get("crop", "Unknown")
    pathogen = rec.get("pathogen", "N/A")
    description = rec.get("description", "No description available.")
    actions = rec.get("actions", [])
    prevention = rec.get("prevention", "No prevention guidelines.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if is_confident:
        status_label = "CONFIRMED (High Confidence >= 60%)"
        diagnosis_header = f"Primary Diagnosis    : {title}"
        action_header = "RECOMMENDED IMMEDIATE ACTIONS:"
        action_body = "\n".join([f"  [{i}] {act}" for i, act in enumerate(actions, start=1)])
        prevention_body = f"  • {prevention}"
    else:
        status_label = "UNCERTAIN / UNCONFIRMED (Low Confidence < 60%)"
        diagnosis_header = f"Top Candidate (Unconfirmed): {title}"
        action_header = "RECOMMENDED GUIDANCE (LOW CONFIDENCE):"
        action_body = (
            "  [!] The AI is not sufficiently confident in this prediction (< 60%).\n"
            "  [!] Do not apply disease-specific treatments based on this unconfirmed prediction.\n"
            "  [!] Please upload a clearer image of a Tomato, Potato, or Pepper leaf under good lighting."
        )
        prevention_body = (
            "  • Ensure the leaf is in focus, well-lit, and from a supported crop (Tomato, Potato, Pepper).\n"
            "  • If symptoms persist, consult your local agricultural extension service for lab diagnosis."
        )

    report_lines = [
        "=" * 60,
        "🌿 AI PLANT HEALTH ASSISTANT - DIAGNOSTIC REPORT",
        "=" * 60,
        f"Generated Timestamp : {timestamp}",
        f"Inference Model      : Plant Disease CNN (128x128)",
        f"Diagnostic Status    : {status_label}",
        f"Target Crop Group    : {crop}",
        diagnosis_header,
        f"Suspected Pathogen   : {pathogen}",
        f"Model Confidence     : {confidence:.2f}% (Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%)",
        f"Leaf Area Affected   : {affected_pct:.2f}% (Estimated)",
        f"Severity Rating      : {severity}",
        "-" * 60,
        "TOP PREDICTION PROBABILITIES:",
    ]

    for rank, (c_name, prob) in enumerate(top3, start=1):
        report_lines.append(f"  {rank}. {c_name.replace('_', ' ')} : {prob:.2f}%")

    report_lines.extend([
        "-" * 60,
        "PATHOLOGY DESCRIPTION (REFERENCE):",
        description,
        "-" * 60,
        action_header,
        action_body,
        "-" * 60,
        "PREVENTION & NEXT STEPS:",
        prevention_body,
        "=" * 60,
        "DISCLAIMER: This diagnostic estimate is generated by AI & Computer",
        "Vision models. For critical crops, consult local agricultural extension.",
        "=" * 60,
    ])

    return "\n".join(report_lines)


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

with st.sidebar:
    st.markdown("### 🌿 Plant Health AI")
    st.caption("AI-powered leaf disease diagnosis & care.")


# ============================================================
# MAIN DASHBOARD INTERFACE (CENTERED)
# ============================================================

# Hero Header Banner
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🌱 AI Plant Health Assistant</div>
    <p class="hero-subtitle">Upload a plant leaf image and the AI will estimate the plant condition.</p>
</div>
""", unsafe_allow_html=True)

# Image Acquisition (Upload Only)
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of an individual leaf (Tomato, Potato, or Bell Pepper)."
)

input_image_rgb = None
image_source_label = ""

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    decoded = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if decoded is not None:
        input_image_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        image_source_label = f"Uploaded: {uploaded_file.name}"


# When an image is ready for analysis
if input_image_rgb is not None:

    # 1. Original Leaf Image View
    st.subheader("🍃 Leaf Inspection")
    st.image(
        input_image_rgb,
        caption=f"{image_source_label}",
        use_container_width=True
    )

    # 2. Validate that the uploaded image is likely a plant leaf
    if not is_likely_leaf_image(input_image_rgb):
        st.divider()
        st.subheader("📊 Analysis Results")
        st.warning(
            "🚫 **Unsupported image detected**\n\n"
            "The uploaded image does not appear to contain a clear plant leaf. "
            "This AI model is designed for **Tomato, Potato, and Bell Pepper leaves** "
            "and cannot reliably diagnose other objects or scenes.\n\n"
            "👉 Please upload a clear, well-lit photo of a single supported plant leaf."
        )
        st.stop()

    # 3. Run AI Prediction & Severity Estimation
    with st.spinner("Analyzing plant leaf condition..."):
        predicted_disease, confidence, top3_preds, latency_ms = predict_disease(
            input_image_rgb
        )

    rec_info = RECOMMENDATIONS.get(predicted_disease, {})
    plant_crop = rec_info.get("crop", "Solanaceous Plant")
    disease_title = rec_info.get("title", predicted_disease.replace("_", " "))
    health_status = rec_info.get("status", "Unknown")

    if health_status == "Healthy":
        affected_pct = 0.0
        severity_level = "None"
    else:
        affected_pct, severity_level = estimate_severity(input_image_rgb)

    # Evaluate confidence threshold guard
    is_confident = confidence >= (CONFIDENCE_THRESHOLD * 100)

    # 3. Diagnosis Summary & Badges
    st.divider()
    st.subheader("📊 Analysis Results")

    if is_confident:
        # High Confidence Path
        if health_status == "Healthy":
            badge_html = '<span class="badge-healthy">🟢 HEALTHY PLANT</span>'
        elif severity_level in ["High", "Severe"]:
            badge_html = '<span class="badge-danger">🔴 DISEASE DETECTED • SEVERE</span>'
        else:
            badge_html = '<span class="badge-warning">🟠 DISEASE DETECTED • MODERATE</span>'

        st.markdown(f"### {disease_title} &nbsp; {badge_html}", unsafe_allow_html=True)
        st.caption(f"**Target Crop:** {plant_crop} &nbsp;•&nbsp; **Pathogen:** {rec_info.get('pathogen', 'N/A')}")
    else:
        # Low Confidence / Out-of-Scope Guard Path
        badge_html = '<span class="badge-warning">⚠️ UNCERTAIN DIAGNOSIS</span>'
        st.markdown(f"### Uncertain Prediction &nbsp; {badge_html}", unsafe_allow_html=True)
        st.caption("The AI confidence is below the reliability threshold (60%).")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Metrics Grid (2x2)
    col1, col2 = st.columns(2)
    with col1:
        if is_confident:
            st.metric("Disease", disease_title)
        else:
            st.metric("Top Candidate", f"{disease_title} (Unconfirmed)")
        
        st.metric(
            "Estimated Affected Area",
            f"{affected_pct:.2f}%"
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%",
            delta="High Certainty" if confidence >= 85 else ("Moderate (>=60%)" if is_confident else "Low (<60%)"),
            delta_color="normal" if is_confident else "inverse"
        )
        st.metric(
            "Estimated Severity",
            severity_level
        )

    st.divider()

    # 5. Top Predictions Probabilities Chart
    st.markdown("##### 📈 Top Prediction Probabilities")
    chart_df = pd.DataFrame({
        "Condition": [c[0].replace("_", " ") for c in top3_preds],
        "Probability (%)": [round(c[1], 2) for c in top3_preds]
    }).sort_values("Probability (%)", ascending=True)

    st.bar_chart(
        chart_df.set_index("Condition"),
        horizontal=True,
        use_container_width=True,
        color="#10b981" if is_confident else "#f59e0b"
    )

    # 6. Confidence Warnings & Guidance Callout
    if not is_confident:
        st.warning(
            "⚠️ **Low-confidence result**\n\n"
            "The AI is not sufficiently confident in this prediction. "
            "The image may be unclear or may not match the crops supported by this model (Tomato, Potato, Pepper).\n\n"
            "👉 Please upload a clear image of a Tomato, Potato, or Pepper leaf under good lighting for a reliable assessment."
        )
    elif confidence >= 90:
        st.success("🎯 **High Confidence**: The model produced a high-confidence prediction.")

    # 7. Agronomic Action & Advisory Center
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🌱 Treatment & Recommendations")

    act_tab1, act_tab2, act_tab3, act_tab4 = st.tabs([
        "💊 Recommended Actions",
        "📖 Disease Description",
        "🛡️ Prevention & Care",
        "📥 Export Report"
    ])

    with act_tab1:
        if is_confident:
            st.markdown(f"#### Recommended Actions for **{disease_title}**")
            actions_list = rec_info.get("actions", [])
            if actions_list:
                for idx, action in enumerate(actions_list, start=1):
                    st.markdown(f"**{idx}.** {action}")
            else:
                st.write("No specific actions listed. Maintain standard crop monitoring.")
        else:
            st.markdown("#### Recommended Guidance")
            st.info(
                "⚠️ **Please upload a clearer image of a supported crop for a more reliable assessment.**\n\n"
                "Because the prediction confidence is below 60%, disease-specific chemical or cultural treatments are not recommended at this time to prevent improper crop management."
            )
            st.markdown("""
            **Next steps:**
            - Verify that the photographed plant is one of the supported crops: **Tomato**, **Potato**, or **Bell Pepper**.
            - Retake the photo with the leaf fully in focus, well-lit by natural light, and against a plain background.
            - If disease symptoms persist, consult a local agricultural extension specialist.
            """)

    with act_tab2:
        if is_confident:
            st.markdown(f"#### Overview: {disease_title}")
            st.write(rec_info.get("description", "No detailed description available."))
            st.markdown(f"**Identified Crop:** {plant_crop}")
            st.markdown(f"**Pathogen:** *{rec_info.get('pathogen', 'N/A')}*")
        else:
            st.markdown("#### Candidate Pathology Overview (Unconfirmed)")
            st.write(
                f"The closest matching condition was **{disease_title}** (Confidence: {confidence:.2f}%), "
                f"but this result is unconfirmed due to low confidence."
            )
            st.write(rec_info.get("description", "No detailed description available."))
            st.caption(f"Host Crop: {plant_crop} | Pathogen: {rec_info.get('pathogen', 'N/A')}")

    with act_tab3:
        st.markdown("#### Prevention & Cultural Practices")
        if is_confident:
            st.write(rec_info.get("prevention", "Practice regular crop rotation, sanitary pruning, and soil testing."))
        else:
            st.write("General health guidelines for Solanaceous crops (Tomato, Potato, Pepper):")
        st.markdown("""
        > **General Best Practices:**
        > - Avoid wetting foliage during watering; water at the base or use drip lines.
        > - Maintain good spacing and airflow between plants.
        > - Sanitize pruning tools regularly.
        """)

    with act_tab4:
        st.markdown("#### 📄 Export Diagnostic Summary")
        st.write("Download a timestamped, structured agronomy summary report for farm records.")

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
            file_name=f"plant_health_report_{predicted_disease}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=False
        )

else:
    # Empty State & Getting Started Section
    st.info("👈 Please **upload a leaf image** above to begin.")

    st.markdown("""
    <div class="guide-box">
        <h3 style="margin-top: 0; color: #ecfdf5;">🚀 Getting Started</h3>
        <p style="color: #cbd5e1;">Follow these steps to analyze your plant leaf:</p>
        <ol style="color: #cbd5e1; line-height: 1.8;">
            <li><strong>Upload a Leaf Image:</strong> Provide a clear JPG, JPEG, or PNG photo of a single Tomato, Potato, or Pepper leaf.</li>
            <li><strong>Automated AI Diagnosis:</strong> The deep learning model will classify the disease and estimate severity and affected area.</li>
            <li><strong>Review Action Plan:</strong> Follow immediate treatment guidelines and download an agronomic diagnostic report.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("AI Plant Health Assistant • CNN + Computer Vision")