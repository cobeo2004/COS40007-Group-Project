import streamlit as st
import os
import cv2  # For video processing, ensure it's in requirements.txt
import time
import pandas as pd
import numpy as np  # For potential direct frame manipulation if needed
from PIL import Image  # For displaying frames
import json  # For history

# --- Session State Initialization ---
if 'video_analysis_results' not in st.session_state:  # For currently run analysis
    st.session_state.video_analysis_results = None
if 'video_prediction_history' not in st.session_state:  # List of video analysis detail dicts
    st.session_state.video_prediction_history = []
if 'active_video_analysis_details' not in st.session_state:  # Dict for current/selected video view
    st.session_state.active_video_analysis_details = None
if 'confirm_delete_video_history' not in st.session_state:  # For delete confirmation
    st.session_state.confirm_delete_video_history = False
if 'video_upload_key_counter' not in st.session_state:  # To reset file_uploader on demand
    st.session_state.video_upload_key_counter = 0

# Assuming 'packages' is in PYTHONPATH or structure allows this import
from packages.utils.model_loader import BoundingBoxModelLoader, SegmentationModelLoader
from packages.video_model_predictor import VideoModelPredictor

# Define paths - adjust as per your project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BBOX_MODEL_PATH = os.path.join(BASE_DIR, "models", "bbox.pt")
SEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "segment.pt")
# For uploaded original videos
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
# For processed frames/videos
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "predictions")
# Directory for history files
HISTORY_DIR = os.path.join(BASE_DIR, "data", "history")
VIDEO_HISTORY_FILE_PATH = os.path.join(
    HISTORY_DIR, "video_prediction_history.json")

# --- Debug: Print Base Paths ---
print(f"[DEBUG-VideoPage] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG-VideoPage] UPLOAD_DIR: {UPLOAD_DIR}")
print(f"[DEBUG-VideoPage] OUTPUT_DIR: {OUTPUT_DIR}")
print(f"[DEBUG-VideoPage] HISTORY_DIR: {HISTORY_DIR}")
print(f"[DEBUG-VideoPage] VIDEO_HISTORY_FILE_PATH: {VIDEO_HISTORY_FILE_PATH}")
print(f"[DEBUG-VideoPage] BBOX_MODEL_PATH: {BBOX_MODEL_PATH}")
print(f"[DEBUG-VideoPage] SEG_MODEL_PATH: {SEG_MODEL_PATH}")

# Create directories if they don't exist
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    print(
        f"[DEBUG-VideoPage] Directories {UPLOAD_DIR}, {OUTPUT_DIR}, and {HISTORY_DIR} ensured.")
except OSError as e:
    st.error(f"Error creating data directories: {e}")
    print(f"[ERROR-VideoPage] Error creating data directories: {e}")

# --- Video History Persistence Functions ---


def load_video_history_from_file():
    if os.path.exists(VIDEO_HISTORY_FILE_PATH):
        try:
            with open(VIDEO_HISTORY_FILE_PATH, 'r') as f:
                history = json.load(f)
                print(
                    f"[DEBUG-VideoPage] Loaded {len(history)} video items from history file.")
                return history
        except json.JSONDecodeError:
            print(
                f"[ERROR-VideoPage] Could not decode JSON from video history file. Starting fresh.")
            return []
        except Exception as e:
            print(
                f"[ERROR-VideoPage] Could not load video history file: {e}. Starting fresh.")
            return []
    return []


def save_video_history_to_file(history_data):
    try:
        with open(VIDEO_HISTORY_FILE_PATH, 'w') as f:
            json.dump(history_data, f, indent=4)
        print(
            f"[DEBUG-VideoPage] Saved {len(history_data)} video items to history file.")
    except Exception as e:
        st.error(f"Error saving video history to file: {e}")
        print(f"[ERROR-VideoPage] Could not save video history file: {e}")


# Load video history into session state at the start if it's empty
if not st.session_state.video_prediction_history:
    loaded_history = load_video_history_from_file()
    if loaded_history:
        st.session_state.video_prediction_history = loaded_history

# Cached function to load models


@st.cache_resource
def load_models_for_video():
    print("[DEBUG-VideoPage] Attempting to load models for video page...")
    try:
        if not os.path.exists(BBOX_MODEL_PATH):
            st.error(f"Bounding Box model not found at: {BBOX_MODEL_PATH}")
            return None
        if not os.path.exists(SEG_MODEL_PATH):
            st.error(f"Segmentation model not found at: {SEG_MODEL_PATH}")
            return None

        bbox_loader = BoundingBoxModelLoader(BBOX_MODEL_PATH)
        seg_loader = SegmentationModelLoader(SEG_MODEL_PATH)
        video_predictor = VideoModelPredictor(bbox_loader, seg_loader)
        print("[DEBUG-VideoPage] Models and VideoModelPredictor loaded successfully.")
        return video_predictor
    except Exception as e:
        st.error(f"Critical error loading models for video: {e}")
        print(f"[CRITICAL ERROR-VideoPage] loading models: {e}")
        return None


# Page configuration
st.set_page_config(page_title="Video Prediction", page_icon="🎬", layout="wide")

# Header
st.title("🎬 Video-based Structural Defect Detection")
st.markdown("Upload a video for analysis. View past analyses from the sidebar. Processed videos can be saved in `data/predictions`.")

# Load models
video_model_predictor = load_models_for_video()

# --- Sidebar: Video Prediction History ---
st.sidebar.subheader("Video Prediction History")
if not video_model_predictor:  # Also check if models loaded before enabling history interaction tied to re-analysis
    st.sidebar.error(
        "Models not loaded. Video analysis & history features disabled.")
elif not st.session_state.video_prediction_history:
    st.sidebar.info("No video predictions made yet, or history file is empty.")
else:
    # Display newest first
    for i, entry in enumerate(reversed(st.session_state.video_prediction_history)):
        # Ensure original_video_filename exists, provide a fallback
        display_name = entry.get(
            'original_video_filename', f"Analysis {entry.get('id', 'N/A')}")
        if st.sidebar.button(f"{entry.get('timestamp', 'N/A')} - {display_name}", key=f"video_history_{entry.get('id', i)}"):
            st.session_state.active_video_analysis_details = entry
            # Clear any results from a direct run
            st.session_state.video_analysis_results = None
            st.session_state.confirm_delete_video_history = False  # Reset delete confirmation
            print(
                f"[DEBUG-VideoPage] Selected video history item: {entry.get('id', 'N/A')}")
            st.rerun()

if video_model_predictor:  # Show ready message only if models are truly ready
    st.sidebar.success("Models ready for video analysis!")

# Delete Video History Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Manage Video History")

if st.session_state.confirm_delete_video_history:
    st.sidebar.warning(
        "Delete all video history and associated video files? This cannot be undone.")
    if st.sidebar.button("Yes, Delete All Video History", type="primary"):
        deleted_files_count = 0
        not_found_count = 0
        error_deleting_count = 0
        history_to_delete = list(st.session_state.video_prediction_history)

        for entry in history_to_delete:
            paths_to_delete = [
                entry.get('original_video_path'), entry.get('processed_video_path')]
            for path in paths_to_delete:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        deleted_files_count += 1
                    except OSError as e:
                        error_deleting_count += 1
                        print(
                            f"[ERROR-VideoPage] Could not delete video file {path}: {e}")
                elif path:
                    not_found_count += 1

        st.session_state.video_prediction_history = []
        st.session_state.active_video_analysis_details = None
        save_video_history_to_file([])  # Save empty list
        st.session_state.confirm_delete_video_history = False

        feedback = f"Video history deletion complete. {deleted_files_count} files deleted."
        if not_found_count > 0:
            feedback += f" {not_found_count} files not found."
        if error_deleting_count > 0:
            feedback += f" Failed to delete {error_deleting_count} files."
        st.sidebar.success(feedback)
        print(f"[INFO-VideoPage] {feedback}")
        st.rerun()

    if st.sidebar.button("Cancel Video History Deletion"):
        st.session_state.confirm_delete_video_history = False
        st.rerun()
# Only show if history exists and models loaded
elif st.session_state.video_prediction_history and video_model_predictor:
    if st.sidebar.button("Delete Video Prediction History"):
        st.session_state.confirm_delete_video_history = True
        st.rerun()
else:
    st.session_state.confirm_delete_video_history = False

# --- Main Page: File Uploader and Analysis ---
uploaded_file = st.file_uploader(
    "Choose a video...",
    type=["mp4", "mov", "avi", "mkv"],
    key=f"video_uploader_{st.session_state.video_upload_key_counter}"
)

if uploaded_file is not None:
    # When a new file is uploaded, clear any selected historical analysis
    # to avoid confusion. The new upload takes precedence for the main interaction area.
    if st.session_state.active_video_analysis_details and \
       st.session_state.active_video_analysis_details.get("original_video_filename") != uploaded_file.name:
        st.session_state.active_video_analysis_details = None  # Clear old history view
        # Potentially rerun if needed, or let the flow continue to analysis button

    st.video(uploaded_file)

    st.subheader("Analysis Parameters")
    col1, col2 = st.columns(2)
    with col1:
        frame_interval = st.slider(
            "Frame sampling interval (frames)", 1, 60, 15, help="Process every Nth frame.")
        detection_threshold = st.slider(
            "Detection confidence threshold", 0.0, 1.0, 0.35, help="Minimum confidence.")

    with col2:
        available_defect_types = ["Crack", "Rust",
                                  "Tower Structure"]  # User's preference
        defect_types_to_detect = st.multiselect(
            "Select defect types to detect",
            options=["All"] + available_defect_types,
            default=["All"]
        )
        save_processed_output = st.checkbox(
            # Default to true for video
            "Save processed video output (to data/predictions)", value=True)

    analyze_button_key = f"analyze_video_btn_{uploaded_file.name}_{uploaded_file.size}"
    if st.button("Analyze Uploaded Video", key=analyze_button_key):
        if video_model_predictor is None:
            st.error("Video model predictor not available.")
        else:
            st.session_state.video_analysis_results = None  # Clear previous run results
            st.session_state.active_video_analysis_details = None  # Clear active history view

            with st.spinner("Analyzing video... This might take a while."):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                unique_timestamp_id = f"{timestamp}_{int(time.time()*1000) % 1000}"

                # Define persistent path for the original uploaded video
                original_video_filename_ext = os.path.splitext(uploaded_file.name)[
                    1]
                persistent_original_video_name = f"{unique_timestamp_id}_orig{original_video_filename_ext}"
                persistent_original_video_path = os.path.join(
                    UPLOAD_DIR, persistent_original_video_name)

                try:
                    with open(persistent_original_video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    print(
                        f"[DEBUG-VideoPage] Uploaded video saved persistently to: {persistent_original_video_path}")

                    st_progress_bar = st.progress(0)
                    def progress_callback(val): st_progress_bar.progress(val)

                    analysis_start_time = time.time()

                    # Determine output directory for processed video if saving
                    processed_output_dir_for_predictor = OUTPUT_DIR if save_processed_output else None

                    detections_by_frame, processed_ind_frames_paths, saved_proc_video_path = video_model_predictor.process_video(
                        # Use the persistently saved original
                        video_path=persistent_original_video_path,
                        frame_interval=frame_interval,
                        detection_threshold=detection_threshold,
                        defect_types=defect_types_to_detect,
                        progress_callback=progress_callback,
                        output_dir=processed_output_dir_for_predictor,
                        save_processed_video=save_processed_output
                    )
                    analysis_duration = time.time() - analysis_start_time
                    st_progress_bar.progress(1.0)

                    current_analysis_data = {
                        "id": unique_timestamp_id,
                        "timestamp": timestamp,
                        "original_video_path": persistent_original_video_path,
                        "processed_video_path": saved_proc_video_path,  # This can be None
                        "detections_by_frame": detections_by_frame,
                        "processed_individual_frames_paths": processed_ind_frames_paths,
                        "original_video_filename": uploaded_file.name,
                        "analysis_duration_sec": analysis_duration,
                        "parameters": {
                            "frame_interval": frame_interval,
                            "detection_threshold": detection_threshold,
                            "defect_types": defect_types_to_detect,
                            "save_output": save_processed_output
                        }
                    }
                    st.session_state.active_video_analysis_details = current_analysis_data

                    # Add to history
                    st.session_state.video_prediction_history.append(
                        current_analysis_data)
                    # Keep last 10
                    st.session_state.video_prediction_history = st.session_state.video_prediction_history[-10:]
                    save_video_history_to_file(
                        st.session_state.video_prediction_history)

                    st.success(
                        f"Video analysis complete in {analysis_duration:.2f} seconds!")
                    if saved_proc_video_path:
                        st.info(
                            f"Processed video saved to: {saved_proc_video_path}")
                    elif save_processed_output:
                        st.warning(
                            "Save output was checked, but no processed video path was returned. It might not have been saved if no detections occurred or an issue arose.")

                    # No need to delete persistent_original_video_path as it's part of history
                except Exception as e:
                    st.error(f"An error occurred during video analysis: {e}")
                    print(f"[ERROR-VideoPage] Analysis error: {e}")
                    # Potentially remove persistent_original_video_path if analysis failed critically before history save
                    if 'persistent_original_video_path' in locals() and os.path.exists(persistent_original_video_path) and \
                       not any(h.get('original_video_path') == persistent_original_video_path for h in st.session_state.video_prediction_history):
                        try:
                            os.remove(persistent_original_video_path)
                            print(
                                f"[DEBUG-VideoPage] Cleaned up orphaned original video due to error: {persistent_original_video_path}")
                        except Exception as e_del:
                            print(
                                f"[WARNING-VideoPage] Could not delete orphaned original video {persistent_original_video_path}: {e_del}")
                    st.session_state.active_video_analysis_details = None  # Reset on critical failure
                finally:
                    st.session_state.video_upload_key_counter += 1
                    st.rerun()

# --- Display Area for Active or Selected Historical Video Analysis ---
active_details_to_display = st.session_state.active_video_analysis_details

if active_details_to_display:
    details = active_details_to_display
    st.subheader(
        f"Analysis Details: {details.get('timestamp', 'N/A')} - {details.get('original_video_filename', 'Unknown Video')}")

    # Display Original and Processed Videos
    col_orig, col_proc = st.columns(2)
    original_video_path = details.get('original_video_path')
    processed_video_path_from_history = details.get('processed_video_path')

    with col_orig:
        st.markdown("#### Original Video")
        if original_video_path and os.path.exists(original_video_path):
            try:
                st.video(original_video_path)
            except Exception as e:
                st.error(
                    f"Could not load original video: {e}. Path: {original_video_path}")
        elif original_video_path:
            st.warning(
                f"Original video file not found at: {original_video_path}. It may have been moved or deleted.")
        else:
            st.info("Original video path not recorded.")

    with col_proc:
        st.markdown("#### Processed Video (with Detections)")
        if processed_video_path_from_history and os.path.exists(processed_video_path_from_history):
            try:
                st.video(processed_video_path_from_history)
            except Exception as e:
                st.error(
                    f"Could not load processed video: {e}. Path: {processed_video_path_from_history}")
        # Path recorded but not found
        elif details.get('parameters', {}).get('save_output') and processed_video_path_from_history:
            st.warning(
                f"Processed video file not found at: {processed_video_path_from_history}. It may have been moved or deleted.")
        # Should have been saved but no path
        elif details.get('parameters', {}).get('save_output'):
            st.info(
                "Processed video was meant to be saved, but is not available. It might have had no detections or failed to save.")
        else:
            st.info(
                "Processed video was not saved (as per analysis settings or no detections).")

    # Tabs for detailed results
    tab_summary, tab_frame_analysis, tab_timeline, tab_output_params = st.tabs([
        "📊 Summary", "🖼️ Frame Analysis", "⏳ Timeline", "💾 Output & Parameters"
    ])

    detections_by_frame_data = details.get('detections_by_frame', [])
    analysis_params = details.get('parameters', {})

    with tab_summary:
        st.markdown("#### Defect Summary")
        total_detected_frames = len(detections_by_frame_data)
        total_defects_found = sum(len(f['detections'])
                                  for f in detections_by_frame_data)
        st.metric("Frames with Detections", total_detected_frames)
        st.metric("Total Defects Reported", total_defects_found)
        if total_defects_found > 0:
            all_detected_types = [det.get(
                'type', 'Unknown') for f_data in detections_by_frame_data for det in f_data['detections']]
            if all_detected_types:
                df_defect_counts = pd.Series(
                    all_detected_types).value_counts().reset_index()
                df_defect_counts.columns = ['Defect Type', 'Count']
                fig = {
                    'data': [{'x': df_defect_counts['Defect Type'], 'y': df_defect_counts['Count'], 'type': 'bar',
                              'marker': {'color': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#FFD166', '#06D6A0'][:len(df_defect_counts)]}}],
                    'layout': {'height': 400, 'margin': {'t': 20, 'b': 100}, 'xaxis': {'title': 'Defect Type'}, 'yaxis': {'title': 'Count'}}
                }
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No defects found matching the criteria.")

    with tab_frame_analysis:
        st.markdown("#### Frame-by-Frame Analysis")
        processed_individual_frames = details.get(
            'processed_individual_frames_paths', [])
        if processed_individual_frames:
            st.markdown("**Saved Processed Frames (Sample):**")
            # Show up to 5
            for frame_path in processed_individual_frames[:min(5, len(processed_individual_frames))]:
                if os.path.exists(frame_path):
                    st.image(frame_path, caption=os.path.basename(
                        frame_path), use_container_width=True)
                else:
                    st.warning(f"Saved frame not found: {frame_path}")

        st.markdown("**Detections per Processed Frame:**")
        if not detections_by_frame_data:
            st.write("No detections logged for any processed frames.")
        else:
            for i, frame_data in enumerate(detections_by_frame_data):
                ts_seconds = frame_data['timestamp_ms'] / 1000.0
                with st.expander(f"Frame {frame_data['frame_number']} (Time: {ts_seconds:.2f}s) - {len(frame_data['detections'])} defects"):
                    for det_idx, det in enumerate(frame_data['detections']):
                        conf_str = f"{det.get('confidence', 0.0):.2f}" if isinstance(
                            det.get('confidence'), float) else "N/A"
                        st.markdown(
                            f"  - Defect {det_idx+1}: Type: `{det.get('type', 'Unknown')}`, Conf: `{conf_str}`")
                        if 'bbox' in det:
                            st.markdown(f"    - BBox: `{det['bbox']}`")

    with tab_timeline:
        st.markdown("#### Detection Timeline")
        if not detections_by_frame_data:
            st.info("No detections to plot.")
        else:
            timeline_data = [{"Time (s)": f['timestamp_ms'] / 1000.0, "Defect Count": len(
                f['detections'])} for f in detections_by_frame_data]
            df_timeline = pd.DataFrame(timeline_data)
            if not df_timeline.empty:
                st.line_chart(df_timeline.set_index("Time (s)"))
            else:
                st.info("Not enough data for timeline.")

    with tab_output_params:
        st.markdown("#### Analysis Parameters & Output")
        st.json(analysis_params)
        if processed_video_path_from_history and os.path.exists(processed_video_path_from_history):
            st.markdown(
                f"**Processed Video Saved To:** `{processed_video_path_from_history}`")
            try:
                with open(processed_video_path_from_history, "rb") as fp:
                    st.download_button("Download Processed Video", fp, os.path.basename(
                        processed_video_path_from_history), "video/mp4")
            except Exception as e:
                st.error(f"Could not offer video download: {e}")
        elif analysis_params.get('save_output'):
            st.warning(
                "Processed video was set to be saved, but no path is available or file not found.")
        else:
            st.info("Processed video was not saved (as per settings).")

        if processed_individual_frames:
            st.markdown("**Saved Individual Processed Frames:**")
            for p_path in processed_individual_frames:
                st.markdown(f"- `{p_path}`")

elif uploaded_file is None and not st.session_state.active_video_analysis_details:
    st.info("Upload a video to begin analysis or select an item from the video history sidebar.")

# Instructions expander
with st.expander("How to use this tool"):
    st.markdown("""
    1. Ensure models are `bbox.pt` and `segment.pt` in `models/`.
    2. Upload a video (MP4, MOV, AVI, MKV).
    3. Adjust parameters: frame interval, confidence threshold, defect types, save processed video.
    4. Click 'Analyze Uploaded Video'.
    5. View results: Original & processed videos side-by-side, plus tabs for Summary, Frame Analysis, Timeline, Output & Parameters.
    6. Past analyses are in the sidebar. Click to reload. Use "Delete Video Prediction History" to clear history and files.
    """)
