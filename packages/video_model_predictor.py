import cv2
import numpy as np
import time
import os
from packages.utils.model_loader import BoundingBoxModelLoader, SegmentationModelLoader
# To reuse single frame prediction logic
from packages.image_model_predictor import ImageModelPredictor


class VideoModelPredictor:
    def __init__(self, bbox_model_loader: BoundingBoxModelLoader, seg_model_loader: SegmentationModelLoader):
        """
        Initializes the VideoModelPredictor.

        Args:
            bbox_model_loader: Loader for the bounding box detection model.
            seg_model_loader: Loader for the segmentation model.
        """
        # We need an instance of ImageModelPredictor to process individual frames
        self.image_predictor = ImageModelPredictor(
            bbox_model_loader, seg_model_loader)
        print("[DEBUG] VideoModelPredictor initialized with ImageModelPredictor.")

    def process_video(self, video_path: str, frame_interval: int, detection_threshold: float, defect_types: list, progress_callback=None, output_dir: str = None, save_processed_video: bool = False):
        """
        Processes a video file to detect defects in its frames.

        Args:
            video_path (str): Path to the input video file.
            frame_interval (int): Interval at which frames are processed (e.g., process every Nth frame).
            detection_threshold (float): Confidence threshold for detections.
            defect_types (list): List of defect types to focus on (e.g., ["Cracks", "Rust"]). "All" means all types.
            progress_callback (function, optional): A function to call with progress updates (0.0 to 1.0).
            output_dir (str, optional): Directory to save processed frames or video. If None, nothing is saved.
            save_processed_video (bool): If True and output_dir is provided, saves the processed video.

        Returns:
            tuple: A tuple containing:
                - all_detections (list): A list of dictionaries, where each dictionary contains
                                         'frame_number', 'timestamp_ms', and 'detections' (list of detection dicts).
                - processed_frames_paths (list): A list of paths to saved processed frames (if output_dir is set).
                - processed_video_path (str): Path to the saved processed video (if save_processed_video is True).
        """
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            raise IOError(f"Could not open video file: {video_path}")

        all_detections_by_frame = []
        processed_frames_paths = []
        processed_video_writer = None
        processed_video_path = None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(
                f"[DEBUG] Created output directory for video processing: {output_dir}")

        if save_processed_video and output_dir:
            base_name = os.path.basename(video_path)
            name, ext = os.path.splitext(base_name)
            # Use mp4 for wider compatibility
            processed_video_filename = f"processed_{name}_{int(time.time())}.mp4"
            processed_video_path = os.path.join(
                output_dir, processed_video_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
            processed_video_writer = cv2.VideoWriter(
                processed_video_path, fourcc, fps, (frame_width, frame_height))
            print(
                f"[DEBUG] Initialized video writer for: {processed_video_path}")

        frame_number = 0
        processed_frame_count = 0
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(
                    f"[DEBUG] Processing frame {frame_number} (Timestamp: {timestamp_ms:.2f}ms)")

                # Use the image_predictor to process the frame
                processed_frame_bgr, frame_detections = self.image_predictor.predict_on_image(
                    frame_bgr.copy())  # Send a copy

                # Filter detections based on threshold and type
                filtered_detections = []
                if frame_detections:
                    for det in frame_detections:
                        # Default to 0 if no confidence
                        conf = det.get('confidence', 0.0)
                        # Ensure confidence is float for comparison
                        try:
                            conf_float = float(
                                conf if conf is not None else 0.0)
                        except ValueError:
                            conf_float = 0.0

                        det_type = det.get('type', 'Unknown')

                        type_match = "All" in defect_types or det_type in defect_types
                        if conf_float >= detection_threshold and type_match:
                            filtered_detections.append(det)

                if filtered_detections:
                    all_detections_by_frame.append({
                        "frame_number": frame_number,
                        "timestamp_ms": timestamp_ms,
                        "detections": filtered_detections,
                        # Could be added if frames are saved individually pre-analysis
                        "original_frame_path": None,
                        "processed_frame_path": None  # Will be updated if frame is saved
                    })
                    print(
                        f"[DEBUG] Frame {frame_number}: Found {len(filtered_detections)} defects after filtering.")

                if output_dir and not save_processed_video:
                    # Save if detections or one of first 5 processed
                    if filtered_detections or (processed_frame_count < 5):
                        frame_filename = f"frame_{frame_number:05d}_{int(timestamp_ms)}.jpg"
                        frame_save_path = os.path.join(
                            output_dir, frame_filename)
                        try:
                            cv2.imwrite(frame_save_path, processed_frame_bgr)
                            processed_frames_paths.append(frame_save_path)
                            # If we saved it, update the record
                            if filtered_detections and all_detections_by_frame:
                                all_detections_by_frame[-1]["processed_frame_path"] = frame_save_path
                            print(
                                f"[DEBUG] Saved processed frame: {frame_save_path}")
                        except Exception as e:
                            print(
                                f"[ERROR] Could not save frame {frame_save_path}: {e}")

                if processed_video_writer:
                    processed_video_writer.write(processed_frame_bgr)

                processed_frame_count += 1

            frame_number += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_number / total_frames)

        cap.release()
        if processed_video_writer:
            processed_video_writer.release()
            print(
                f"[DEBUG] Released video writer. Processed video saved to: {processed_video_path}")

        if not all_detections_by_frame and not processed_video_path and not processed_frames_paths:
            print("[INFO] No defects found or frames processed according to criteria.")

        print(
            f"[INFO] Video processing complete. Total frames processed: {processed_frame_count}. Detections logged for {len(all_detections_by_frame)} frames.")
        return all_detections_by_frame, processed_frames_paths, processed_video_path
