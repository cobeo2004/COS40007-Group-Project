from packages.live_model_predictor import LiveModelPredictor
from packages.utils.model_loader import BoundingBoxModelLoader, SegmentationModelLoader
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import json  # For history persistence

st.set_page_config(page_title="Webcam Prediction",
                   page_icon="📹", layout="wide")

# --- Package Imports ---
# Adjust paths as necessary if your project structure is different
# Assuming 'packages' is in PYTHONPATH or structure allows this import

# --- Constants and Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BBOX_MODEL_PATH = os.path.join(BASE_DIR, "models", "bbox.pt")
SEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "segment.pt")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "snapshots")
LIVE_PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "live_predictions")
# For history JSON files
HISTORY_DIR = os.path.join(BASE_DIR, "data", "history")
WEBCAM_HISTORY_FILE_PATH = os.path.join(
    HISTORY_DIR, "webcam_snapshot_history.json")

# Ensure all necessary data directories exist
try:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(LIVE_PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)  # Ensure history directory exists
    print(f"[DEBUG-WebcamPage] Snapshot directory {SNAPSHOT_DIR} ensured.")
    print(
        f"[DEBUG-WebcamPage] Live predictions directory {LIVE_PREDICTIONS_DIR} ensured.")
    print(f"[DEBUG-WebcamPage] History directory {HISTORY_DIR} ensured.")
except OSError as e:
    st.error(f"Error creating data directories: {e}")
    print(f"[ERROR-WebcamPage] Error creating data directories: {e}")

AVAILABLE_CLASSES = ["Crack", "Rust", "Tower Structure"]

# --- History Persistence Functions ---


def load_snapshot_history_from_file():
    if os.path.exists(WEBCAM_HISTORY_FILE_PATH):
        try:
            with open(WEBCAM_HISTORY_FILE_PATH, 'r') as f:
                history = json.load(f)
                print(
                    f"[DEBUG-WebcamPage] Loaded {len(history)} snapshots from history file.")
                return history
        except json.JSONDecodeError:
            print(
                f"[ERROR-WebcamPage] Could not decode JSON from snapshot history. Starting fresh.")
            return []
        except Exception as e:
            print(
                f"[ERROR-WebcamPage] Could not load snapshot history: {e}. Starting fresh.")
            return []
    return []


def save_snapshot_history_to_file(snapshot_data):
    try:
        with open(WEBCAM_HISTORY_FILE_PATH, 'w') as f:
            json.dump(snapshot_data, f, indent=4)
        print(
            f"[DEBUG-WebcamPage] Saved {len(snapshot_data)} snapshots to history file.")
    except Exception as e:
        st.error(f"Error saving snapshot history: {e}")
        print(f"[ERROR-WebcamPage] Could not save snapshot history: {e}")


# --- Session State Initialization ---
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "fps" not in st.session_state:
    st.session_state.fps = 0.0
if "snapshots" not in st.session_state:
    # Load from file if empty
    st.session_state.snapshots = load_snapshot_history_from_file()
if "live_predictor" not in st.session_state:
    st.session_state.live_predictor = None
if "models_loaded_webcam" not in st.session_state:
    st.session_state.models_loaded_webcam = False
if "auto_save_live_frames" not in st.session_state:
    st.session_state.auto_save_live_frames = False
if "confirm_delete_snapshot_history" not in st.session_state:  # For delete confirmation
    st.session_state.confirm_delete_snapshot_history = False

# If snapshots were loaded, ensure they are sorted (e.g., by timestamp_display if available or filename)
# This ensures newest appear first if that's the desired display logic later.
# For now, loading order is preserved, which is usually chronological from the file.

# --- Model Loading ---


@st.cache_resource
def load_models_for_webcam():
    print("[DEBUG-WebcamPage] Attempting to load models for webcam...")
    try:
        if not os.path.exists(BBOX_MODEL_PATH):
            st.error(f"Bounding Box model not found: {BBOX_MODEL_PATH}")
            print(
                f"[ERROR-WebcamPage] BBox model not found: {BBOX_MODEL_PATH}")
            return None
        if not os.path.exists(SEG_MODEL_PATH):
            # Segmentation model might be optional for some features
            print(
                f"[WARNING-WebcamPage] Segmentation model not found or not configured: {SEG_MODEL_PATH}")
            # For now, let's allow proceeding without it, but LiveModelPredictor should handle it
            seg_loader = None
        else:
            seg_loader = SegmentationModelLoader(SEG_MODEL_PATH)

        bbox_loader = BoundingBoxModelLoader(BBOX_MODEL_PATH)
        predictor = LiveModelPredictor(
            bbox_loader, seg_loader)  # seg_loader can be None
        print("[DEBUG-WebcamPage] LiveModelPredictor initialized.")
        st.session_state.models_loaded_webcam = True
        return predictor
    except Exception as e:
        st.error(f"Critical error loading models for webcam: {e}")
        print(f"[CRITICAL ERROR-WebcamPage] loading models: {e}")
        st.session_state.models_loaded_webcam = False
        return None


# Load models if not already loaded (e.g., on first run or if state was reset)
if st.session_state.live_predictor is None:
    st.session_state.live_predictor = load_models_for_webcam()

# --- Page Configuration & Header ---
st.title("📹 Real-time Webcam Structural Defect Detection")
st.markdown(
    "View your webcam feed and see real-time structural defect analysis. Settings are on the sidebar.")

if not st.session_state.models_loaded_webcam:
    st.sidebar.error(
        "Models not loaded. Predictions may be limited or unavailable.")
elif st.session_state.live_predictor:
    st.sidebar.success("Models loaded and ready!")
else:
    st.sidebar.warning("Model predictor not available. Please check logs.")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Detection Settings")
    confidence_threshold = st.slider(
        "Confidence threshold for detections", 0.0, 1.0, 0.35, step=0.05)

    st.subheader("Defect Types to Detect")
    # Using multiselect for target classes
    selected_target_classes = st.multiselect(
        "Select defect types:",
        options=AVAILABLE_CLASSES,
        default=AVAILABLE_CLASSES  # Default to all
    )

    st.subheader("Display Options")
    show_bounding_boxes = st.checkbox("Show bounding boxes", value=True)
    show_confidence_scores = st.checkbox("Show confidence scores", value=True)
    highlight_detected_defects = st.checkbox(
        "Highlight detected defects", value=True)

    st.subheader("Persistence")
    st.checkbox("Automatically save frames with detections", key="auto_save_live_frames",
                help=f"Saves frames with any detections to the '{LIVE_PREDICTIONS_DIR}' folder. This can consume disk space quickly.")

    st.markdown("---")
    st.subheader("Manage Snapshot History")
    if st.session_state.confirm_delete_snapshot_history:
        st.sidebar.warning(
            "Delete all snapshot history and associated image files? This cannot be undone.")
        if st.sidebar.button("Yes, Delete All Snapshot History", type="primary"):
            deleted_files_count = 0
            not_found_count = 0
            error_deleting_count = 0
            history_to_delete = list(
                st.session_state.snapshots)  # Iterate over a copy

            for entry in history_to_delete:
                path = entry.get('image_path')
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        deleted_files_count += 1
                    except OSError as e_del:
                        error_deleting_count += 1
                        print(
                            f"[ERROR-WebcamPage] Could not delete snapshot file {path}: {e_del}")
                elif path:  # Path was recorded but file doesn't exist
                    not_found_count += 1

            st.session_state.snapshots = []
            save_snapshot_history_to_file([])  # Save empty list to JSON
            st.session_state.confirm_delete_snapshot_history = False

            feedback = f"Snapshot history deletion complete. {deleted_files_count} files deleted."
            if not_found_count > 0:
                feedback += f" {not_found_count} files not found."
            if error_deleting_count > 0:
                feedback += f" Failed to delete {error_deleting_count} files."
            st.sidebar.success(feedback)
            print(f"[INFO-WebcamPage] {feedback}")
            st.rerun()

        if st.sidebar.button("Cancel Snapshot History Deletion"):
            st.session_state.confirm_delete_snapshot_history = False
            st.rerun()
    elif st.session_state.snapshots:  # Only show delete button if there is history
        if st.sidebar.button("Delete Snapshot History"):
            st.session_state.confirm_delete_snapshot_history = True
            st.rerun()
    else:
        # Ensure it's false if no history
        st.session_state.confirm_delete_snapshot_history = False

# --- Main Content Area ---
col1, col2 = st.columns([3, 1])  # Adjusted column ratio for better layout

with col1:
    st.subheader("Camera Feed")
    camera_image_buffer = st.camera_input(
        "Enable webcam to start analysis", key="webcam_feed")

detections_this_frame = []  # To store detections from the current frame

if camera_image_buffer is not None and st.session_state.live_predictor:
    try:
        pil_image = Image.open(camera_image_buffer).convert('RGB')
        frame_rgb = np.array(pil_image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Prepare detection settings for the predictor
        current_detection_settings = {
            "target_classes": selected_target_classes,
            "confidence_threshold": confidence_threshold,
            "show_bounding_boxes": show_bounding_boxes,
            "show_confidence": show_confidence_scores,
            "highlight_defects": highlight_detected_defects
        }

        # Get predictions
        processed_frame_bgr, detections_this_frame = st.session_state.live_predictor.predict_on_frame(
            frame_bgr,
            target_classes=selected_target_classes,
            confidence_threshold=confidence_threshold,
            show_bounding_boxes=show_bounding_boxes,
            show_confidence=show_confidence_scores,
            highlight_defects=highlight_detected_defects
        )

        # Convert processed frame back to RGB for Streamlit display
        processed_frame_rgb = cv2.cvtColor(
            processed_frame_bgr, cv2.COLOR_BGR2RGB)

        with col1:  # Display processed image in the first column
            st.image(processed_frame_rgb,
                     caption="Live feed with detections", use_container_width=True)

            # Update FPS and frame count
            st.session_state.frame_count += 1
            elapsed_time = time.time() - st.session_state.start_time
            if elapsed_time > 1:  # Update FPS every second to avoid erratic display
                st.session_state.fps = st.session_state.frame_count / elapsed_time
                # Reset for next calculation period to get more real-time FPS
                st.session_state.start_time = time.time()
                st.session_state.frame_count = 0

            # Snapshot button
            if st.button("📸 Take Snapshot & Save Details", key="snapshot_btn"):
                timestamp_str = time.strftime("%Y%m%d-%H%M%S")
                # Add milliseconds for higher uniqueness if snapshots are taken rapidly
                ms_timestamp = int(time.time() * 1000)
                snapshot_filename = f"snapshot_{timestamp_str}_{ms_timestamp % 1000}.png"
                snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
                try:
                    # Save the processed frame (RGB for PIL)
                    Image.fromarray(processed_frame_rgb).save(snapshot_path)
                    new_snapshot_details = {
                        "image_path": snapshot_path,
                        "filename": snapshot_filename,
                        # Ensure it's a list
                        "detections": list(detections_this_frame),
                        "timestamp_display": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "timestamp_unix": ms_timestamp,  # For potential sorting
                        "settings_used": current_detection_settings.copy()
                    }
                    st.session_state.snapshots.append(new_snapshot_details)
                    # Keep snapshots sorted by unix timestamp, newest first for display
                    st.session_state.snapshots.sort(
                        key=lambda x: x.get('timestamp_unix', 0), reverse=True)
                    # Limit history size (e.g., last 50 snapshots)
                    SNAPSHOT_HISTORY_LIMIT = 50
                    st.session_state.snapshots = st.session_state.snapshots[:SNAPSHOT_HISTORY_LIMIT]
                    save_snapshot_history_to_file(
                        st.session_state.snapshots)  # Save updated history
                    st.success(
                        f"Snapshot saved as {snapshot_filename} in {SNAPSHOT_DIR}")
                except Exception as e:
                    st.error(f"Error saving snapshot: {e}")
                    print(f"[ERROR-WebcamPage] Saving snapshot: {e}")

            # Auto-save frame if option is enabled and detections are present
            if st.session_state.auto_save_live_frames and detections_this_frame:
                try:
                    current_time_ms = int(time.time() * 1000)
                    auto_save_filename = f"live_pred_{current_time_ms}.png"
                    auto_save_path = os.path.join(
                        LIVE_PREDICTIONS_DIR, auto_save_filename)
                    Image.fromarray(processed_frame_rgb).save(auto_save_path)
                    # This print statement is for debugging; avoid st.write or st.info here to prevent clutter
                    print(
                        f"[DEBUG-WebcamPage] Auto-saved frame with detections to {auto_save_path}")
                except Exception as e:
                    print(f"[ERROR-WebcamPage] Error auto-saving frame: {e}")
                    # Optionally, provide a less intrusive notification if auto-save fails, e.g., a small warning icon.

    except Exception as e:
        with col1:
            st.error(f"Error processing webcam frame: {e}")
        print(f"[ERROR-WebcamPage] Processing frame: {e}")

elif camera_image_buffer is None:
    with col1:
        st.info("Webcam not active or no image received. Please enable your webcam.")
elif not st.session_state.live_predictor:
    with col1:
        st.error(
            "Live predictor is not initialized. Cannot process webcam feed. Check model loading status.")

# --- Column 2: Detection Results, Metrics, and Snapshots ---
with col2:
    st.subheader("Live Analysis")

    if camera_image_buffer is not None and st.session_state.live_predictor:
        # Display current frame's detection results
        if detections_this_frame:
            st.markdown("**Detected Defects (Current Frame):**")
            # Create a simple display for current detections
            for det in detections_this_frame:
                st.markdown(
                    f"- **{det.get('type', 'N/A')}**: {det.get('confidence', 0.0):.2f} conf.")
        else:
            st.info("No defects detected in the current frame based on settings.")

        # Display metrics
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("FPS", f"{st.session_state.fps:.1f}")
        metric_col2.metric("Detected Objects", len(detections_this_frame))
    else:
        st.info("Enable webcam and ensure models are loaded to see live analysis.")

    # Display Snapshots
    if st.session_state.snapshots:
        st.markdown("--- ")  # Visual separator
        st.subheader("Saved Snapshots History")
        # Display snapshots sorted by timestamp (newest first due to sort on save)
        snapshot_options = {f"{s.get('timestamp_display', 'N/A')} - {s.get('filename', 'Unknown')}": i
                            for i, s in enumerate(st.session_state.snapshots)}

        if not snapshot_options:
            st.info("No snapshots in history to display.")
        else:
            selected_snapshot_display_key = st.selectbox(
                "View snapshot from history:", options=list(snapshot_options.keys()))

            if selected_snapshot_display_key:
                selected_snapshot_index = snapshot_options[selected_snapshot_display_key]
                selected_snap_details = st.session_state.snapshots[selected_snapshot_index]
                try:
                    if not os.path.exists(selected_snap_details["image_path"]):
                        st.error(
                            f"Snapshot image not found: {selected_snap_details['image_path']}. It may have been deleted.")
                    else:
                        snap_img = Image.open(
                            selected_snap_details["image_path"])
                        st.image(
                            snap_img, caption=f"Snapshot: {selected_snap_details['filename']}", use_container_width=True)
                        st.markdown("**Detections in this snapshot:**")
                        if selected_snap_details["detections"]:
                            for det in selected_snap_details["detections"]:
                                bbox_str = f", BBox: {det['bbox']}" if 'bbox' in det else ""
                                st.markdown(
                                    f"- **{det.get('type', 'N/A')}**: {det.get('confidence', 0.0):.2f}{bbox_str}")
                        else:
                            st.info(
                                "No defects were recorded for this snapshot based on its settings.")
                        with st.expander("Settings used for this snapshot"):
                            st.json(selected_snap_details.get(
                                "settings_used", {}))
                except FileNotFoundError:  # Should be caught by os.path.exists, but as a fallback
                    st.error(
                        f"Snapshot image file not found: {selected_snap_details['image_path']}.")
                except Exception as e:
                    st.error(f"Error displaying snapshot: {e}")
    else:
        st.info("No snapshots taken yet or history is empty.")

# --- Instructions and Warnings ---
st.markdown("--- ")
with st.expander("ℹ️ How to Use & Requirements"):
    st.markdown("""
    1.  **Enable Webcam**: Click the "Enable webcam" button above.
    2.  **Grant Permissions**: Your browser will ask for permission to use the camera. Please allow it.
    3.  **Adjust Settings**: Use the sidebar to set the confidence threshold, defect types, and display options.
    4.  **View Live Feed**: The left panel shows the live camera feed with detections.
    5.  **Analyze Results**: The right panel shows live FPS, detected objects, and any defects found in the current frame.
    6.  **Take Snapshots**: Click "📸 Take Snapshot & Save Details" to manually save the current frame. Its details are added to the history.
    7.  **View History**: Select past snapshots from the "View snapshot from history" dropdown in the right panel.
    8.  **Auto-Save**: If "Automatically save frames with detections" is checked in the sidebar, frames with any detected defects will be saved to `data/live_predictions/` (these are not part of the browsable snapshot history).
    9.  **Manage History**: Use "Delete Snapshot History" in the sidebar to clear all saved snapshots and their records.

    **Requirements:**
    - A functioning webcam connected to your computer.
    - Browser permissions granted for camera access.
    - Adequate lighting for optimal detection quality.
    - Ensure models (`bbox.pt`, `segment.pt`) are in the `models/` directory.
    """)

st.warning("""
⚠️ **Accuracy Note:** Real-time webcam analysis provides immediate feedback but may have lower accuracy compared to dedicated Image or Video analysis pages. Factors like camera movement, lighting variations, and processing constraints can affect performance. For critical inspections, use high-quality, stable images or videos with the other tools.

💾 **Storage Note:** Enabling "Automatically save frames with detections" can consume disk space rapidly, especially with frequent detections. Regularly check and manage the `data/live_predictions/` folder. Snapshot history also consumes space.
""")
