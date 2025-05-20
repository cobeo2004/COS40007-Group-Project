import os
import cv2
import numpy as np
from .utils.model_loader import BoundingBoxModelLoader, SegmentationModelLoader
# Assuming ModelLoaderImageType is not strictly needed for single image processing,
# or it's handled within the model loaders themselves.
# from interfaces.model_loader_interface import ModelLoaderImageType


class ImageModelPredictor:
    def __init__(self, bbox_model: BoundingBoxModelLoader, seg_model: SegmentationModelLoader):
        self.bbox_model = bbox_model
        self.seg_model = seg_model

    def predict_on_image(self, image_np: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Run inference with both models on a single image and return the processed image
        and a list of detections.

        Args:
            image_np (np.ndarray): The input image as a NumPy array (BGR format).

        Returns:
            tuple[np.ndarray, list]: A tuple containing:
                - result_img (np.ndarray): The image with predictions drawn on it (BGR format).
                - detections (list): A list of detected objects/features.
        """
        if image_np is None:
            raise ValueError("Input image cannot be None")

        # Load models - these are already initialized in __init__
        # bbox_model = self.bbox_model
        # seg_model = self.seg_model

        # Run inference with both models
        # Ensure your model's predict method can take a numpy array directly
        # and does not rely on image_path.
        # You might need to adjust your BoundingBoxModelLoader and SegmentationModelLoader
        # if their predict methods expect file paths.
        bbox_results = self.bbox_model.predict(image_np, save=False, verbose=False)
        seg_results = self.seg_model.predict(image_np, save=False, verbose=False)

        result_img_bgr = image_np.copy()
        detections_summary = []

        # Process bounding box detections (Crack and Rust)
        if bbox_results and len(bbox_results[0].boxes) > 0:
            boxes = bbox_results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                detection_info = {"type": "Unknown BBox", "confidence": conf, "class_id": cls, "bbox": [x1, y1, x2, y2]}

                if conf > 0.25:  # Confidence threshold
                    # Class 0 is Crack (red color)
                    if cls == 0:
                        color = (0, 0, 255)  # BGR format: red
                        label = f"Crack {conf:.2f}"
                        detection_info["type"] = "Crack"
                    # Class 1 is Rust (blue color)
                    elif cls == 1: # Assuming class 1 is Rust based on original code
                        color = (255, 0, 0)  # BGR format: blue
                        label = f"Rust {conf:.2f}"
                        detection_info["type"] = "Rust"
                    else:
                        color = (0, 255, 255) # Yellow for other classes
                        label = f"Class {cls} {conf:.2f}"


                    # Draw bounding box
                    cv2.rectangle(result_img_bgr, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    cv2.putText(result_img_bgr, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    detections_summary.append(detection_info)


        # Process segmentation detections (Tower Structure)
        if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
            masks = seg_results[0].masks

            # Create an overlay for the segmentation
            overlay = result_img_bgr.copy()
            tower_structure_detected = False

            for i, mask_tensor in enumerate(masks.data):
                # Convert mask tensor to numpy array and resize to image dimensions
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np, (result_img_bgr.shape[1], result_img_bgr.shape[0]))

                # Apply the mask (assuming segmentation is for tower structure - green)
                # Check if the mask actually covers a significant area
                if np.sum(mask_resized > 0.5) > 0: # Ensure mask is not empty
                    overlay[mask_resized > 0.5] = overlay[mask_resized > 0.5] * 0.7 + np.array([0, 255, 0]) * 0.3 # Green
                    tower_structure_detected = True

            if tower_structure_detected:
                # Add the overlay to the result image
                cv2.addWeighted(overlay, 0.7, result_img_bgr, 0.3, 0, result_img_bgr)

                # Add label for Tower Structure
                cv2.putText(result_img_bgr, "Tower Structure", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Green
                detections_summary.append({"type": "Tower Structure", "segmentation_applied": True})

        return result_img_bgr, detections_summary

    # The run_inference_and_combine method is now replaced by predict_on_image
    # for single image processing. If you still need batch processing,
    # you can keep a similar method or adapt it to use predict_on_image.
    # For now, I will comment it out to avoid confusion.
    """
    def run_inference_and_combine(self, test_images_dir, output_dir):
        # ... (original batch processing code) ...
        # This method would now ideally iterate through images and call self.predict_on_image(img_np)
        # and then handle saving.
        pass
    """
