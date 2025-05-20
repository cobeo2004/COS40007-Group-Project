import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(page_title="Webcam Prediction", page_icon="📹")

# Header
st.title("📹 Real-time Webcam Structural Defect Detection")
st.markdown("Use your webcam for real-time structural defect analysis")

def process_frame(frame, detection_settings):
    """
    Process a frame with detection settings
    """
    # Create a copy for drawing
    processed_frame = frame.copy()

    # Simulate some detections for demonstration
    detections = []
    detection_threshold = detection_settings.get('detection_threshold', 0.5)

    # Simulate cracks (red)
    if detection_settings.get('detect_cracks', True):
        if np.random.random() > 0.7:  # 30% chance of detecting a crack
            confidence = np.random.random()
            if confidence > detection_threshold:
                x, y, w, h = np.random.randint(0, processed_frame.shape[1]-100), np.random.randint(0, processed_frame.shape[0]-100), 100, 20
                if detection_settings.get('show_bounding_boxes', True):
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if detection_settings.get('show_confidence', True):
                    cv2.putText(processed_frame, f"Crack: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                detections.append({"type": "Crack", "confidence": confidence, "location": "Random position"})

    # Simulate corrosion (green)
    if detection_settings.get('detect_corrosion', True):
        if np.random.random() > 0.8:  # 20% chance of detecting corrosion
            confidence = np.random.random()
            if confidence > detection_threshold:
                x, y, w, h = np.random.randint(0, processed_frame.shape[1]-150), np.random.randint(0, processed_frame.shape[0]-150), 150, 150
                if detection_settings.get('show_bounding_boxes', True):
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if detection_settings.get('show_confidence', True):
                    cv2.putText(processed_frame, f"Corrosion: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detections.append({"type": "Corrosion", "confidence": confidence, "location": "Random position"})

    return processed_frame, detections

# Initialize session state
if "start_time" not in st.session_state:
    st.session_state["start_time"] = time.time()
if "frame_count" not in st.session_state:
    st.session_state["frame_count"] = 0
if "fps" not in st.session_state:
    st.session_state["fps"] = 0
if "snapshots" not in st.session_state:
    st.session_state["snapshots"] = []

# Sidebar for settings
with st.sidebar:
    st.header("Detection Settings")
    detection_threshold = st.slider("Detection threshold", 0.0, 1.0, 0.5)

    st.subheader("Defect Types")
    detect_cracks = st.checkbox("Detect cracks", value=True)
    detect_corrosion = st.checkbox("Detect corrosion", value=True)
    detect_spalling = st.checkbox("Detect spalling", value=True)
    detect_delamination = st.checkbox("Detect delamination", value=False)

    st.subheader("Display Options")
    show_bounding_boxes = st.checkbox("Show bounding boxes", value=True)
    show_confidence = st.checkbox("Show confidence scores", value=True)
    highlight_defects = st.checkbox("Highlight defects", value=True)

# Collect all detection settings
detection_settings = {
    "detection_threshold": detection_threshold,
    "detect_cracks": detect_cracks,
    "detect_corrosion": detect_corrosion,
    "detect_spalling": detect_spalling,
    "detect_delamination": detect_delamination,
    "show_bounding_boxes": show_bounding_boxes,
    "show_confidence": show_confidence,
    "highlight_defects": highlight_defects
}

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Camera Feed")

    # Use Streamlit's native camera input
    camera_image = st.camera_input("Enable webcam")

    if camera_image is not None:
        # Convert the image to a numpy array
        img = Image.open(camera_image)
        img_array = np.array(img)

        # Process the frame
        processed_frame, detections = process_frame(img_array, detection_settings)

        # Display the processed frame
        st.image(processed_frame, caption="Live feed with defect detection", channels="RGB")

        # Update metrics
        st.session_state["frame_count"] += 1
        elapsed_time = time.time() - st.session_state["start_time"]
        if elapsed_time > 0:
            st.session_state["fps"] = st.session_state["frame_count"] / elapsed_time

        # Take snapshot button
        if st.button("Take Snapshot"):
            st.session_state["snapshots"].append({
                "image": processed_frame.copy(),
                "detections": detections.copy(),
                "timestamp": time.strftime("%H:%M:%S")
            })
            st.success(f"Snapshot taken at {time.strftime('%H:%M:%S')}")

with col2:
    st.subheader("Detection Results")

    if camera_image is not None:
        # Display detection results
        if detections:
            st.markdown("### Detected Defects:")
            for detection in detections:
                st.markdown(f"- **{detection['type']}**: {detection['confidence']:.2f} confidence")
        else:
            st.info("No defects detected in current frame.")

        # Display metrics
        col1, col2 = st.columns(2)
        col1.metric("FPS", f"{st.session_state['fps']:.1f}")
        col2.metric("Detected Objects", len(detections) if 'detections' in locals() else 0)

    # Display snapshots
    if st.session_state["snapshots"]:
        st.markdown("---")
        st.subheader("Saved Snapshots")

        # Get the most recent snapshot
        latest_snapshot = st.session_state["snapshots"][-1]

        # Display the snapshot
        st.image(latest_snapshot["image"],
                caption=f"Snapshot taken at {latest_snapshot['timestamp']}",
                channels="RGB")

        # Show total snapshots taken
        total_snapshots = len(st.session_state["snapshots"])
        if total_snapshots > 1:
            st.caption(f"{total_snapshots} snapshots taken. Showing most recent.")

# Camera requirements note
st.markdown("---")
with st.expander("Camera Requirements"):
    st.markdown("""
    This feature requires webcam access. Please ensure:

    1. Your browser has permission to access the camera
    2. Your camera is functioning properly
    3. You have adequate lighting for best detection results

    For best results, hold the camera steady and ensure the structural element is clearly visible.
    """)

# Warning about accuracy
st.warning("""
Please note that real-time detection accuracy may be lower than image-based analysis due to factors like:
- Camera movement
- Lighting conditions
- Frame processing limitations
For critical inspections, we recommend using the Image or Video analysis tools with high-quality footage.
""")
