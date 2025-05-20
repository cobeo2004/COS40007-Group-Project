import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import json

# --- Session State Initialization ---
# This MUST be at the top, after imports, before any other code that might use session_state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []  # List of analysis detail dicts
if 'active_analysis_details' not in st.session_state:
    st.session_state.active_analysis_details = None # Dict for current/selected view
if 'confirm_delete_history' not in st.session_state: # For delete confirmation
    st.session_state.confirm_delete_history = False

# Make sure 'packages' is in the Python path or use relative imports
# This might require adjusting your project structure or PYTHONPATH
# For simplicity, assuming utils and interfaces are accessible
from packages.utils.model_loader import BoundingBoxModelLoader, SegmentationModelLoader
from packages.image_model_predictor import ImageModelPredictor

# Define paths for models and data
# Corrected paths using __file__ to be relative to the current script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BBOX_MODEL_PATH = os.path.join(BASE_DIR, "models", "bbox.pt")
SEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "segment.pt")

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "predictions")
HISTORY_DIR = os.path.join(BASE_DIR, "data", "history") # Directory for history file
HISTORY_FILE_PATH = os.path.join(HISTORY_DIR, "prediction_history.json")

# --- Debug: Print Base Paths ---
print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] UPLOAD_DIR: {UPLOAD_DIR}")
print(f"[DEBUG] OUTPUT_DIR: {OUTPUT_DIR}")
print(f"[DEBUG] HISTORY_DIR: {HISTORY_DIR}")
print(f"[DEBUG] HISTORY_FILE_PATH: {HISTORY_FILE_PATH}")
print(f"[DEBUG] BBOX_MODEL_PATH: {BBOX_MODEL_PATH}")
print(f"[DEBUG] SEG_MODEL_PATH: {SEG_MODEL_PATH}")
print(f"[DEBUG] Initializing script. Confirm delete flag: {st.session_state.confirm_delete_history}")

# Create directories if they don't exist
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True) # Ensure history directory exists
    print(f"[DEBUG] Directories {UPLOAD_DIR}, {OUTPUT_DIR}, and {HISTORY_DIR} ensured.")
except OSError as e:
    st.error(f"Error creating data directories: {e}")
    print(f"[ERROR] Error creating data directories: {e}")

# --- History Persistence Functions ---
def load_history_from_file():
    if os.path.exists(HISTORY_FILE_PATH):
        try:
            with open(HISTORY_FILE_PATH, 'r') as f:
                history = json.load(f)
                print(f"[DEBUG] Loaded {len(history)} items from history file.")
                return history
        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode JSON from history file: {HISTORY_FILE_PATH}. Starting fresh.")
            return []
        except Exception as e:
            print(f"[ERROR] Could not load history file {HISTORY_FILE_PATH}: {e}. Starting fresh.")
            return []
    return []

def save_history_to_file(history_data):
    try:
        with open(HISTORY_FILE_PATH, 'w') as f:
            json.dump(history_data, f, indent=4)
        print(f"[DEBUG] Saved {len(history_data)} items to history file.")
    except Exception as e:
        st.error(f"Error saving history to file: {e}")
        print(f"[ERROR] Could not save history file {HISTORY_FILE_PATH}: {e}")

# Load history into session state at the start if it's empty
# This ensures that on the first run of a session, history is loaded from file.
# Subsequent runs in the same session will use the existing st.session_state.prediction_history.
if not st.session_state.prediction_history: # Only load if session history is currently empty
    # This check is important to prevent reloading from file on every rerun within a session,
    # which would overwrite in-memory changes made during that session until the next save.
    # However, for page reloads which reset session_state, this will trigger.
    # A more robust way for true persistence might involve checking a flag or timestamp.
    # For now, this loads history if the session_state list is empty, common after a page refresh.
    loaded_history = load_history_from_file()
    if loaded_history: # Check if anything was actually loaded
        st.session_state.prediction_history = loaded_history

# Cached function to load models
@st.cache_resource
def load_models():
    print("[DEBUG] Attempting to load models...")
    try:
        # Ensure model paths are correct and files exist
        if not os.path.exists(BBOX_MODEL_PATH):
            st.error(f"Bounding Box model not found at: {BBOX_MODEL_PATH}")
            print(f"[ERROR] Bounding Box model not found at: {BBOX_MODEL_PATH}")
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
st.set_page_config(page_title="Image Prediction", page_icon="🖼️", layout="wide")

# Header
st.title("🖼️ Image-based Structural Defect Detection")
st.markdown("Upload an image or select a past analysis from the history in the sidebar.")

# Load models and instantiate predictor
bbox_model_loader, seg_model_loader = load_models()
image_predictor = None
if bbox_model_loader and seg_model_loader:
    try:
        image_predictor = ImageModelPredictor(bbox_model_loader, seg_model_loader)
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
        if st.sidebar.button(f"{entry.get('timestamp', 'N/A')} - {entry.get('original_filename', 'Unknown')}", key=f"history_{entry.get('id', i)}"):
            st.session_state.active_analysis_details = entry
            st.session_state.confirm_delete_history = False # Reset delete confirmation
            print(f"[DEBUG] Selected history item: {entry.get('id', 'N/A')}")
            st.rerun() # Explicitly rerun to update the main display

if image_predictor:
    st.sidebar.success("Models ready!")

# Delete History Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Manage History")

if st.session_state.confirm_delete_history:
    st.sidebar.warning("Are you sure you want to delete all history and associated files? This cannot be undone.")
    if st.sidebar.button("Yes, Delete All History", type="primary"):
        print("[DEBUG] Confirm Delete All History button clicked.")
        deleted_files_count = 0
        not_found_count = 0
        error_deleting_count = 0

        # Create a copy for iteration as we might modify the original list (though here we clear it)
        history_to_delete = list(st.session_state.prediction_history)

        for entry in history_to_delete:
            paths_to_delete = [entry.get('input_image_path'), entry.get('output_image_path')]
            for path in paths_to_delete:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        deleted_files_count += 1
                        print(f"[DEBUG] Deleted file: {path}")
                    except OSError as e:
                        error_deleting_count += 1
                        print(f"[ERROR] Could not delete file {path}: {e}")
                elif path: # Path was recorded but file doesn't exist
                    not_found_count +=1
                    print(f"[DEBUG] File not found for deletion (already deleted or never saved?): {path}")

        st.session_state.prediction_history = []
        st.session_state.active_analysis_details = None
        save_history_to_file([]) # Save empty list to the JSON file
        st.session_state.confirm_delete_history = False # Reset confirmation flag

        # Provide feedback
        feedback_message = f"History deletion complete. {deleted_files_count} files deleted."
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
elif st.session_state.prediction_history: # Only show delete button if there is history
    if st.sidebar.button("Delete Prediction History"):
        st.session_state.confirm_delete_history = True
        print("[DEBUG] Delete Prediction History button clicked. Set confirm_delete_history to True.")
        st.rerun()
else: # No history, no need for delete button/confirmation state
    st.session_state.confirm_delete_history = False

# --- Main Page: File Uploader and Analysis Display ---

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if image_predictor is None:
        st.error("Models are not loaded. Cannot analyze image.")
    else:
        # Process new upload immediately if "Analyze Image" is clicked
        # Unique key for the analyze button using file name and upload time to avoid conflicts
        analyze_button_key = f"analyze_btn_{uploaded_file.name}_{uploaded_file.id if hasattr(uploaded_file, 'id') else 'no_id'}"
        if st.button("Analyze Uploaded Image", key=analyze_button_key):
            print(f"[DEBUG] 'Analyze Uploaded Image' button clicked for {uploaded_file.name}")
            st.session_state.confirm_delete_history = False # Reset delete confirmation if an analysis is run
            with st.spinner("Analyzing for structural defects..."):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Add a small random element or microsecond to timestamp for history key if very fast re-uploads occur
                unique_timestamp_id = f"{timestamp}_{int(time.time()*1000)%1000}"
                input_filename = f"{unique_timestamp_id}_{uploaded_file.name}"
                input_image_path = os.path.join(UPLOAD_DIR, input_filename)
                print(f"[DEBUG] Input image path: {input_image_path}")

                try:
                    with open(input_image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    print(f"[DEBUG] Uploaded image saved to: {input_image_path}")
                except IOError as e:
                    st.error(f"Error saving uploaded image: {e}")
                    print(f"[ERROR] Saving uploaded image: {e}")
                    st.session_state.active_analysis_details = None
                except Exception as e: # Catch any other unexpected error during file save
                    st.error(f"Unexpected error saving uploaded image: {e}")
                    print(f"[ERROR] Unexpected error saving input: {e}")
                    st.session_state.active_analysis_details = None

                # Proceed with prediction even if input saving failed, to test prediction logic
                # In a real app, you might stop if input_image_path is crucial for records
                pil_image = Image.open(uploaded_file).convert('RGB')
                original_image_rgb = np.array(pil_image)
                original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)

                try:
                    print(f"[DEBUG] Calling image_predictor.predict_on_image...")
                    processed_image_bgr, detections = image_predictor.predict_on_image(original_image_bgr)
                    print(f"[DEBUG] Prediction successful. Detections: {len(detections) if detections else 0}")

                    output_filename = f"pred_{input_filename}"
                    output_image_path = os.path.join(OUTPUT_DIR, output_filename)
                    print(f"[DEBUG] Output image path: {output_image_path}")
                    try:
                        cv2.imwrite(output_image_path, processed_image_bgr)
                        print(f"[DEBUG] Processed image saved to: {output_image_path}")
                    except cv2.error as e: # More specific error for OpenCV
                        st.error(f"OpenCV Error saving processed image: {e}")
                        print(f"[ERROR] OpenCV cv2.imwrite: {e}")
                    except Exception as e: # General error for saving output
                        st.error(f"Error saving processed image: {e}")
                        print(f"[ERROR] Saving processed image: {e}")

                    current_analysis = {
                        "id": unique_timestamp_id,
                        "timestamp": timestamp,
                        "input_image_path": input_image_path,
                        "output_image_path": output_image_path,
                        "detections": detections,
                        "original_filename": uploaded_file.name
                    }
                    st.session_state.active_analysis_details = current_analysis
                    print(f"[DEBUG] Set active_analysis_details: {current_analysis['id']}")

                    # Update history in session state
                    history_updated = False
                    if not any(h['id'] == unique_timestamp_id for h in st.session_state.prediction_history):
                        st.session_state.prediction_history.append(current_analysis)
                        st.session_state.prediction_history = st.session_state.prediction_history[-10:] # Keep history to last 10
                        history_updated = True
                        print(f"[DEBUG] Added to prediction_history. New length: {len(st.session_state.prediction_history)}")

                    # Save updated history to file
                    if history_updated: # Only save if something actually changed
                        save_history_to_file(st.session_state.prediction_history)

                    st.success("Analysis complete!")
                    st.rerun() # Explicitly rerun to update history and main display
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    print(f"[ERROR] During prediction: {e}")
                    st.session_state.active_analysis_details = None

# --- Display Area for Active or Selected Historical Analysis ---
if st.session_state.active_analysis_details:
    details = st.session_state.active_analysis_details
    # Make sure details dictionary has the expected keys before trying to access them
    analysis_id = details.get('id', 'N/A')
    analysis_timestamp = details.get('timestamp', 'N/A')
    original_filename = details.get('original_filename', 'Unknown')
    input_image_path = details.get('input_image_path')
    output_image_path = details.get('output_image_path')
    current_detections = details.get('detections', [])

    print(f"[DEBUG] Displaying active_analysis_details: {analysis_id}")
    st.markdown(f"### Analysis: {analysis_timestamp} - {original_filename}")

    input_exists = input_image_path and os.path.exists(input_image_path)
    output_exists = output_image_path and os.path.exists(output_image_path)

    if not input_exists:
        st.error(f"Input image not found: {input_image_path}. It may have been moved, deleted, or failed to save.")
    if not output_exists:
        st.error(f"Output image not found: {output_image_path}. It may have been moved, deleted, or failed to save.")

    if input_exists and output_exists:
        try:
            original_img_pil = Image.open(input_image_path)
            processed_img_pil = Image.open(output_image_path)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_img_pil, caption=original_filename, use_column_width=True)
            with col2:
                st.subheader("Processed Image with Detections")
                st.image(processed_img_pil, caption="Processed Detections", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading images for display: {e}")
            print(f"[ERROR] Loading images for display: {e}")

    st.subheader("Detection Summary")
    if current_detections:
        for detection in current_detections:
            det_info = f"- Type: {detection.get('type', 'N/A')}"
            if 'confidence' in detection:
                conf = detection['confidence']
                det_info += f", Confidence: {conf:.2f}" if isinstance(conf, float) else f", Confidence: {conf}"
            if 'bbox' in detection:
                det_info += f", BBox: {detection['bbox']}"
            st.markdown(det_info)
    else:
        st.info("No specific defects reported for this image.")
elif uploaded_file is None:
    # This message shows when the app starts and no file is uploaded yet, and no history is selected.
    st.info("Upload an image to begin analysis or select an item from the history sidebar.")

# Instructions expander
with st.expander("How to use this tool"):
    st.markdown("""
    1. Ensure models are correctly placed: `bbox.pt` and `segment.pt` in the `models/` directory relative to the project root.
    2. Upload an image (JPG, JPEG, PNG).
    3. Click 'Analyze Uploaded Image'.
    4. View results. Uploaded & processed images are saved in `data/uploads/` & `data/predictions/`. History is in `data/history/`.
    5. Past analyses from this session appear in the sidebar. Click to view them again. Reloading the page will load saved history.
    6. Use the "Delete Prediction History" button in the sidebar to clear all history and associated image files.
    """)
