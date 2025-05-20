import os
import cv2
import numpy as np
from .utils.model_loader import BoundingBoxModelLoader, SegmentationModelLoader
# Assuming ModelLoaderImageType is not strictly needed for single image processing,
# or it's handled within the model loaders themselves.
# from interfaces.model_loader_interface import ModelLoaderImageType

# Define class names consistently
CRACK_CLASS_NAME = "Crack"
RUST_CLASS_NAME = "Rust"
TOWER_STRUCTURE_CLASS_NAME = "Tower Structure"

# Mapping from class IDs from your bbox model to class names
# Adjust this if your model has different class IDs/names
BBOX_CLASS_MAP = {
    0: CRACK_CLASS_NAME,  # Assuming class 0 is Crack
    1: RUST_CLASS_NAME    # Assuming class 1 is Rust
}

class ImageModelPredictor:
    def __init__(self, bbox_model: BoundingBoxModelLoader, seg_model: SegmentationModelLoader):
        self.bbox_model = bbox_model
        self.seg_model = seg_model

    def predict_on_image(self, image_np: np.ndarray, target_classes: list[str] | None = None, confidence_threshold: float = 0.25) -> tuple[np.ndarray, list]:
        """
        Run inference with both models on a single image and return the processed image
        and a list of detections.

        Args:
            image_np (np.ndarray): The input image as a NumPy array (BGR format).
            target_classes (list[str] | None): A list of class names to target. If None, all classes are targeted.
            confidence_threshold (float): The confidence threshold for bounding box detections.

        Returns:
            tuple[np.ndarray, list]: A tuple containing:
                - result_img (np.ndarray): The image with predictions drawn on it (BGR format).
                - detections (list): A list of detected objects/features.
        """
        if image_np is None:
            raise ValueError("Input image cannot be None")

        # If target_classes is an empty list, it implies nothing should be targeted.
        # If None, it implies all default classes should be targeted.
        # For clarity, let's treat an empty list as "target nothing specific from user input",
        # but our logic will often check "if target_classes is None or specific_class in target_classes".
        # An empty target_classes list will effectively mean only things not specifically filtered out will run.
        # If target_classes is an empty list, then `specific_class in target_classes` will always be false.

        bbox_results = None
        seg_results = None

        # Decide whether to run bbox model based on target_classes
        # Run if target_classes is None (all) or if any of Crack/Rust are targeted
        should_run_bbox = target_classes is None or any(cls_name in target_classes for cls_name in [CRACK_CLASS_NAME, RUST_CLASS_NAME])

        if should_run_bbox:
            print(f"[Predictor] Running BBox model for targets: {target_classes or 'All'}")
            bbox_results = self.bbox_model.predict(image_np, save=False, verbose=False, conf=confidence_threshold)
        else:
            print(f"[Predictor] Skipping BBox model based on target_classes: {target_classes}")

        # Decide whether to run seg model
        should_run_seg = target_classes is None or TOWER_STRUCTURE_CLASS_NAME in target_classes
        if should_run_seg:
            print(f"[Predictor] Running Segmentation model for targets: {target_classes or 'All'}")
            seg_results = self.seg_model.predict(image_np, save=False, verbose=False)
        else:
            print(f"[Predictor] Skipping Segmentation model based on target_classes: {target_classes}")

        result_img_bgr = image_np.copy()
        detections_summary = []

        # Process bounding box detections
        if bbox_results and len(bbox_results[0].boxes) > 0:
            boxes = bbox_results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                detected_class_name = BBOX_CLASS_MAP.get(class_id) # Get name from map

                # Filter by target_classes
                if detected_class_name and (target_classes is None or detected_class_name in target_classes):
                    # The model.predict(conf=...) should ideally handle filtering by confidence.
                    # If not, the check below is a fallback. For Ultralytics, `conf` in predict applies it.
                    if conf >= confidence_threshold: # Apply threshold check here if not done by model.predict
                        detection_info = {"type": detected_class_name, "confidence": conf, "class_id": class_id, "bbox": [x1, y1, x2, y2]}
                        color = (0, 255, 255) # Default color (yellow for BGR)
                        if detected_class_name == CRACK_CLASS_NAME:
                            color = (0, 0, 255)  # Red for Crack
                        elif detected_class_name == RUST_CLASS_NAME:
                            color = (255, 0, 0)  # Blue for Rust

                        label = f"{detected_class_name} {conf:.2f}"
                        cv2.rectangle(result_img_bgr, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(result_img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        detections_summary.append(detection_info)
                    else:
                        print(f"[Predictor] Skipped {detected_class_name} (conf: {conf:.2f} < threshold: {confidence_threshold:.2f})")
                elif detected_class_name is None:
                    print(f"[Predictor] Warning: Unknown class ID {class_id} from bbox model.")
                # else: (implicitly) class was detected but not in target_classes, so it's skipped

        # Process segmentation detections
        if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
            # Filter for Tower Structure based on target_classes
            if target_classes is None or TOWER_STRUCTURE_CLASS_NAME in target_classes:
                masks = seg_results[0].masks
                overlay = result_img_bgr.copy()
                tower_structure_mask_applied = False

                for i, mask_tensor in enumerate(masks.data):
                    mask_np = mask_tensor.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (result_img_bgr.shape[1], result_img_bgr.shape[0]))
                    if np.sum(mask_resized > 0.5) > 0:
                        overlay[mask_resized > 0.5] = overlay[mask_resized > 0.5] * 0.7 + np.array([0, 255, 0]) * 0.3 # Green for tower
                        tower_structure_mask_applied = True

                if tower_structure_mask_applied:
                    cv2.addWeighted(overlay, 0.7, result_img_bgr, 0.3, 0, result_img_bgr)
                    cv2.putText(result_img_bgr, TOWER_STRUCTURE_CLASS_NAME, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # For segmentation, confidence is often intrinsic to the mask or not provided per-instance like bbox.
                    # We assign a high mock confidence if the structure is detected and targeted.
                    detections_summary.append({"type": TOWER_STRUCTURE_CLASS_NAME, "segmentation_applied": True, "confidence": 1.0})
            # else: (implicitly) Tower Structure was not targeted, so segmentation is skipped

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
