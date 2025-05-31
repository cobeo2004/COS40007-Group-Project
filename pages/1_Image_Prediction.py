import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import json
import pandas as pd

# --- Session State Initialization ---
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []  # List of analysis detail dicts
if "active_analysis_details" not in st.session_state:
    st.session_state.active_analysis_details = None  # Dict for current/selected view
if "confirm_delete_history" not in st.session_state:  # For delete confirmation
    st.session_state.confirm_delete_history = False
if "selected_target_classes" not in st.session_state:  # For class filter persistence
    st.session_state.selected_target_classes = [
        "Crack",
        "Rust",
        "Tower Structure",
    ]  # Default to all
if "confidence_threshold" not in st.session_state:  # For confidence threshold
    st.session_state.confidence_threshold = 0.25  # Default threshold

from packages.utils.model_loader import BoundingBoxModelLoader, SegmentationModelLoader
from packages.image_model_predictor import ImageModelPredictor

# Define paths for models and data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BBOX_MODEL_PATH = os.path.join(BASE_DIR, "models", "bbox.pt")
SEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "segment.pt")

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "predictions")
# Directory for history file
HISTORY_DIR = os.path.join(BASE_DIR, "data", "history")
HISTORY_FILE_PATH = os.path.join(HISTORY_DIR, "prediction_history.json")

CRACK_CLASS_NAME = "Crack"
RUST_CLASS_NAME = "Rust"
TOWER_STRUCTURE_CLASS_NAME = "Tower Structure"
AVAILABLE_CLASSES = [CRACK_CLASS_NAME,
                     RUST_CLASS_NAME, TOWER_STRUCTURE_CLASS_NAME]

# --- Debug: Print Base Paths ---
print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] UPLOAD_DIR: {UPLOAD_DIR}")
print(f"[DEBUG] OUTPUT_DIR: {OUTPUT_DIR}")
print(f"[DEBUG] HISTORY_DIR: {HISTORY_DIR}")
print(f"[DEBUG] HISTORY_FILE_PATH: {HISTORY_FILE_PATH}")
print(f"[DEBUG] BBOX_MODEL_PATH: {BBOX_MODEL_PATH}")
print(f"[DEBUG] SEG_MODEL_PATH: {SEG_MODEL_PATH}")
print(
    f"[DEBUG] Initializing script. Confirm delete flag: {st.session_state.confirm_delete_history}"
)
print(
    f"[DEBUG] Initial selected_target_classes: {st.session_state.selected_target_classes}"
)
print(
    f"[DEBUG] Initial confidence_threshold: {st.session_state.confidence_threshold}")

# Create directories if they don't exist
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)  # Ensure history directory exists
    print(
        f"[DEBUG] Directories {UPLOAD_DIR}, {OUTPUT_DIR}, and {HISTORY_DIR} ensured.")
except OSError as e:
    st.error(f"Error creating data directories: {e}")
    print(f"[ERROR] Error creating data directories: {e}")


# --- History Persistence Functions ---
def load_history_from_file():
    if os.path.exists(HISTORY_FILE_PATH):
        try:
            with open(HISTORY_FILE_PATH, "r") as f:
                history = json.load(f)
                print(
                    f"[DEBUG] Loaded {len(history)} items from history file.")
                return history
        except json.JSONDecodeError:
            print(
                f"[ERROR] Could not decode JSON from history file: {HISTORY_FILE_PATH}. Starting fresh."
            )
            return []
        except Exception as e:
            print(
                f"[ERROR] Could not load history file {HISTORY_FILE_PATH}: {e}. Starting fresh."
            )
            return []
    return []


def save_history_to_file(history_data):
    try:
        with open(HISTORY_FILE_PATH, "w") as f:
            json.dump(history_data, f, indent=4)
        print(f"[DEBUG] Saved {len(history_data)} items to history file.")
    except Exception as e:
        st.error(f"Error saving history to file: {e}")
        print(f"[ERROR] Could not save history file {HISTORY_FILE_PATH}: {e}")


if (
    not st.session_state.prediction_history
):  # Only load if session history is currently empty
    loaded_history = load_history_from_file()
    if loaded_history:  # Check if anything was actually loaded
        st.session_state.prediction_history = loaded_history


# Cached function to load models
@st.cache_resource
def load_models():
    print("[DEBUG] Attempting to load models...")
    try:
        # Ensure model paths are correct and files exist
        if not os.path.exists(BBOX_MODEL_PATH):
            st.error(f"Bounding Box model not found at: {BBOX_MODEL_PATH}")
            print(
                f"[ERROR] Bounding Box model not found at: {BBOX_MODEL_PATH}")
            return None, None
        if not os.path.exists(SEG_MODEL_PATH):
            st.error(f"Segmentation model not found at: {SEG_MODEL_PATH}")
            print(f"[ERROR] Segmentation model not found at: {SEG_MODEL_PATH}")
            return None, None

        bbox_model_loader = BoundingBoxModelLoader(BBOX_MODEL_PATH)
        seg_model_loader = SegmentationModelLoader(SEG_MODEL_PATH)
        print("[DEBUG] Models loaded successfully.")
        return bbox_model_loader, seg_model_loader
    except Exception as e:
        st.error(f"Critical error loading models: {e}")
        print(f"[CRITICAL ERROR] loading models: {e}")
        return None, None


# Page configuration
st.set_page_config(page_title="Image Prediction",
                   page_icon="🖼️", layout="wide")

# Header
st.title("🖼️ Image-based Structural Defect Detection")
st.markdown(
    "Upload an image, set filters, and analyze. Or select a past analysis from history (sidebar)."
)

# Load models and instantiate predictor
bbox_model_loader, seg_model_loader = load_models()
image_predictor = None
if bbox_model_loader and seg_model_loader:
    try:
        image_predictor = ImageModelPredictor(
            bbox_model_loader, seg_model_loader)
        # st.sidebar.success("Models ready!") # Moved to after history display
    except Exception as e:
        st.sidebar.error(f"Predictor init error: {e}")
        print(f"[ERROR] Predictor init error: {e}")
elif not (bbox_model_loader and seg_model_loader):
    st.sidebar.error("Models not loaded. Prediction disabled.")

# --- Sidebar: Prediction History ---
st.sidebar.subheader("Prediction History")
if not st.session_state.prediction_history:
    st.sidebar.info("No predictions made yet, or history file is empty.")
else:
    # Display newest first
    for i, entry in enumerate(reversed(st.session_state.prediction_history)):
        if st.sidebar.button(
            f"{entry.get('timestamp', 'N/A')} - {entry.get('original_filename', 'Unknown')}",
            key=f"history_{entry.get('id', i)}",
        ):
            st.session_state.active_analysis_details = entry
            st.session_state.confirm_delete_history = False
            st.session_state.selected_target_classes = entry.get(
                "targeted_classes", AVAILABLE_CLASSES
            )
            st.session_state.confidence_threshold = entry.get(
                "confidence_threshold", 0.25
            )
            st.rerun()  # Explicitly rerun to update the main display

if image_predictor:
    st.sidebar.success("Models ready!")

# Delete History Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Manage History")

if st.session_state.confirm_delete_history:
    st.sidebar.warning(
        "Are you sure you want to delete all history and associated files? This cannot be undone."
    )
    if st.sidebar.button("Yes, Delete All History", type="primary"):
        print("[DEBUG] Confirm Delete All History button clicked.")
        deleted_files_count = 0
        not_found_count = 0
        error_deleting_count = 0

        # Create a copy for iteration as we might modify the original list (though here we clear it)
        history_to_delete = list(st.session_state.prediction_history)

        for entry in history_to_delete:
            paths_to_delete = [
                entry.get("input_image_path"),
                entry.get("output_image_path"),
            ]
            for path in paths_to_delete:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        deleted_files_count += 1
                        print(f"[DEBUG] Deleted file: {path}")
                    except OSError as e:
                        error_deleting_count += 1
                        print(f"[ERROR] Could not delete file {path}: {e}")
                elif path:  # Path was recorded but file doesn't exist
                    not_found_count += 1
                    print(
                        f"[DEBUG] File not found for deletion (already deleted or never saved?): {path}"
                    )

        st.session_state.prediction_history = []
        st.session_state.active_analysis_details = None
        save_history_to_file([])  # Save empty list to the JSON file
        st.session_state.confirm_delete_history = False  # Reset confirmation flag

        # Provide feedback
        feedback_message = (
            f"History deletion complete. {deleted_files_count} files deleted."
        )
        if not_found_count > 0:
            feedback_message += f" {not_found_count} files were not found (may have been previously deleted)."
        if error_deleting_count > 0:
            feedback_message += f" Failed to delete {error_deleting_count} files (check permissions or logs)."
        st.sidebar.success(feedback_message)
        print(f"[INFO] {feedback_message}")
        st.rerun()

    if st.sidebar.button("Cancel"):
        st.session_state.confirm_delete_history = False
        st.rerun()
elif st.session_state.prediction_history:  # Only show delete button if there is history
    if st.sidebar.button("Delete Prediction History"):
        st.session_state.confirm_delete_history = True
        print(
            "[DEBUG] Delete Prediction History button clicked. Set confirm_delete_history to True."
        )
        st.rerun()
else:
    st.session_state.confirm_delete_history = False

# --- Main Page: File Uploader, Filters, and Analysis ---

# File uploader
uploaded_file = st.file_uploader(
    "1. Choose an image...", type=["jpg", "jpeg", "png"])

# Conditionally display filters and threshold
if uploaded_file is not None or st.session_state.active_analysis_details is not None:
    st.markdown("**2. Set Prediction Filters**")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        st.multiselect(
            "Defect types to detect:",
            options=AVAILABLE_CLASSES,
            default=st.session_state.selected_target_classes,
            key="selected_target_classes",
        )
    with filter_col2:
        st.slider(
            "Confidence threshold for detections:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            key="confidence_threshold",
        )

if uploaded_file is not None:
    if image_predictor is None:
        st.error("Models are not loaded. Cannot analyze image.")
    else:
        analyze_button_key = (
            f"analyze_btn_{uploaded_file.name if uploaded_file else 'no_file'}"
        )
        if st.button("3. Analyze Uploaded Image", key=analyze_button_key):
            st.session_state.confirm_delete_history = False

            selected_classes_for_prediction = st.session_state.selected_target_classes
            current_confidence_threshold = st.session_state.confidence_threshold
            print(
                f"[DEBUG] Analyzing with target classes: {selected_classes_for_prediction}, Threshold: {current_confidence_threshold}"
            )

            with st.spinner("Analyzing for structural defects..."):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                unique_timestamp_id = f"{timestamp}_{int(time.time()*1000) % 1000}"
                input_filename = f"{unique_timestamp_id}_{uploaded_file.name}"
                input_image_path = os.path.join(UPLOAD_DIR, input_filename)
                try:
                    with open(input_image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                except IOError as e:
                    st.error(f"Error saving uploaded image: {e}")
                    st.session_state.active_analysis_details = None
                except Exception as e:
                    st.error(f"Unexpected error saving uploaded image: {e}")
                    st.session_state.active_analysis_details = None

                pil_image = Image.open(uploaded_file).convert("RGB")
                original_image_rgb = np.array(pil_image)
                original_image_bgr = cv2.cvtColor(
                    original_image_rgb, cv2.COLOR_RGB2BGR)

                try:
                    processed_image_bgr, detections = image_predictor.predict_on_image(
                        original_image_bgr,
                        target_classes=selected_classes_for_prediction,
                        confidence_threshold=current_confidence_threshold,
                    )
                    output_filename = f"pred_{input_filename}"
                    output_image_path = os.path.join(
                        OUTPUT_DIR, output_filename)
                    try:
                        cv2.imwrite(output_image_path, processed_image_bgr)
                    except cv2.error as e:
                        st.error(f"OpenCV Error saving processed image: {e}")
                    except Exception as e:
                        st.error(f"Error saving processed image: {e}")

                    current_analysis = {
                        "id": unique_timestamp_id,
                        "timestamp": timestamp,
                        "input_image_path": input_image_path,
                        "output_image_path": output_image_path,
                        "detections": detections,
                        "original_filename": uploaded_file.name,
                        "targeted_classes": selected_classes_for_prediction,
                        "confidence_threshold": current_confidence_threshold,
                    }
                    st.session_state.active_analysis_details = current_analysis
                    history_updated = False
                    if not any(
                        h["id"] == unique_timestamp_id
                        for h in st.session_state.prediction_history
                    ):
                        st.session_state.prediction_history.append(
                            current_analysis)
                        st.session_state.prediction_history = (
                            st.session_state.prediction_history[-10:]
                        )
                        history_updated = True
                    if history_updated:
                        save_history_to_file(
                            st.session_state.prediction_history)
                    st.success("Analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    print(f"[ERROR] Prediction error during analysis: {e}")
                    st.session_state.active_analysis_details = None

# --- Display Area for Active or Selected Historical Analysis ---
if st.session_state.active_analysis_details:
    details = st.session_state.active_analysis_details
    analysis_id = details.get("id", "N/A")
    analysis_timestamp = details.get("timestamp", "N/A")
    original_filename = details.get("original_filename", "Unknown")
    input_image_path = details.get("input_image_path")
    output_image_path = details.get("output_image_path")
    current_detections = details.get("detections", [])
    targeted_classes_display = details.get(
        "targeted_classes", AVAILABLE_CLASSES
    )
    confidence_threshold_display = details.get(
        "confidence_threshold", 0.25
    )

    st.markdown(
        f"### Analysis Results: {analysis_timestamp} - {original_filename}")
    st.caption(
        f"Detections shown for: **{', '.join(targeted_classes_display) if targeted_classes_display else 'None'}** | Confidence Threshold Used: **{confidence_threshold_display:.2f}**"
    )

    input_exists = input_image_path and os.path.exists(input_image_path)
    output_exists = output_image_path and os.path.exists(output_image_path)

    if not input_exists:
        st.error(
            f"Input image not found: {input_image_path}. It may have been moved, deleted, or failed to save."
        )
    if not output_exists:
        st.error(
            f"Output image not found: {output_image_path}. It may have been moved, deleted, or failed to save."
        )

    if input_exists and output_exists:
        try:
            original_img_pil = Image.open(input_image_path)
            processed_img_pil = Image.open(output_image_path)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                if input_exists:
                    try:
                        st.image(
                            original_img_pil,
                            caption=original_filename,
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error loading original image: {e}")
                else:
                    st.warning(f"Original image not found: {input_image_path}")
            with col2:
                st.subheader("Processed Image with Detections")
                if output_exists:
                    try:
                        st.image(
                            processed_img_pil,
                            caption="Processed Detections",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Error loading processed image: {e}")
                else:
                    st.warning(
                        f"Processed image not found: {output_image_path}")
        except Exception as e:
            st.error(f"Error loading images for display: {e}")
            print(f"[ERROR] Loading images for display: {e}")

    st.subheader("Detection Summary")
    if current_detections:
        # Create DataFrame for charts
        detection_data = []
        for i, det in enumerate(current_detections):
            confidence = det.get("confidence")
            try:
                confidence_float = float(
                    confidence) if confidence is not None else 0.0
            except ValueError:
                confidence_float = 0.0
            detection_data.append(
                {
                    "Detection": f"{det.get('type', 'Unknown')} #{i+1}",
                    "Type": det.get("type", "Unknown"),
                    "Confidence": confidence_float,
                }
            )
        df_detections = pd.DataFrame(detection_data)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Confidence Levels per Detection**")
            if (
                not df_detections.empty
                and "Confidence" in df_detections.columns
                and df_detections["Confidence"].sum() > 0
            ):
                chart_df_confidence = df_detections[
                    df_detections["Confidence"] > 0
                ].set_index("Detection")[["Confidence"]]
                if not chart_df_confidence.empty:
                    st.bar_chart(
                        chart_df_confidence.sort_values(
                            by="Confidence", ascending=False
                        ),
                        height=400,
                    )
                else:
                    st.info(
                        "No detections with confidence > 0 to display in chart.")
            else:
                st.info("No applicable confidence data to plot.")

        with chart_col2:
            st.markdown("**Defect Type Counts**")
            if not df_detections.empty and "Type" in df_detections.columns:
                type_counts = df_detections["Type"].value_counts(
                ).reset_index()
                type_counts.columns = ["Type", "Count"]
                colors = ["#FF6B6B", "#4ECDC4",
                          "#45B7D1", "#96CEB4", "#FFEEAD"]
                if not type_counts.empty:
                    fig = {
                        "data": [
                            {
                                "x": type_counts["Type"],
                                "y": type_counts["Count"],
                                "type": "bar",
                                "marker": {"color": colors[: len(type_counts)]},
                            }
                        ],
                        "layout": {"height": 400, "margin": {"t": 10}},
                    }
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No defect types to count.")
            else:
                st.info("No type data for defect counts chart.")
        st.markdown("--- --- --- --- ---")
        st.subheader("Detailed Detection Summary (Scrollable)")
        with st.container(
            height=300
        ):
            for detection in current_detections:
                det_info = f"- **Type**: {detection.get('type', 'N/A')}"
                if "confidence" in detection:
                    conf = detection["confidence"]
                    confidence_str = (
                        f"{conf:.2f}" if isinstance(conf, float) else str(conf)
                    )
                    det_info += f", **Confidence**: {confidence_str}"
                if "bbox" in detection:
                    det_info += f", **BBox**: {detection['bbox']}"
                if (
                    "class_id" in detection
                ):
                    det_info += f", **Class ID**: {detection['class_id']}"
                st.markdown(det_info)
    else:
        st.info("No specific defects reported for this image based on current filters.")
elif uploaded_file is None:
    st.info(
        "Upload an image to begin analysis or select an item from the history sidebar."
    )

with st.expander("How to use this tool"):
    st.markdown(
        """
    1. Ensure models are correctly placed: `bbox.pt` and `segment.pt` in the `models/` directory relative to the project root.
    2. Upload an image (JPG, JPEG, PNG).
    3. Set defect type filters and confidence threshold in the main area below the uploader.
    4. Click 'Analyze Uploaded Image'.
    5. View results. Uploaded & processed images are saved in `data/uploads/` & `data/predictions/`. History is in `data/history/`.
    6. Past analyses from the sidebar show results based on the filters/threshold active *at the time of their analysis*. The current filter/threshold controls *new* analyses.
    7. Use the "Delete Prediction History" button in the sidebar to clear all history and associated image files.
    """
    )
