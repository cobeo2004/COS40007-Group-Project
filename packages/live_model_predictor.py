import cv2
import numpy as np
import os

# Assuming BoundingBoxModelLoader and SegmentationModelLoader are accessible
# from .utils.model_loader import BoundingBoxModelLoader, SegmentationModelLoader
# For now, placeholder until actual model integration
# For demonstration, let's use placeholder model loaders if the real ones aren't ready
# In a real scenario, these would be properly imported and used.

class LiveModelPredictor:
    def __init__(self, bbox_model_loader, seg_model_loader):
        """
        Initializes the LiveModelPredictor with loaded models.
        Args:
            bbox_model_loader: An instance of BoundingBoxModelLoader.
            seg_model_loader: An instance of SegmentationModelLoader.
        """
        self.bbox_model = bbox_model_loader
        self.seg_model = seg_model_loader
        # Placeholder for class names if needed by your models
        self.class_names = {0: "Crack", 1: "Rust", 2: "Tower Structure"} # Example
        print("[DEBUG-LivePredictor] LiveModelPredictor initialized.")
        if not self.bbox_model:
            print("[WARNING-LivePredictor] Bounding Box Model Loader not provided.")
        if not self.seg_model:
            print("[WARNING-LivePredictor] Segmentation Model Loader not provided.")


    def predict_on_frame(self, frame_bgr, target_classes=None, confidence_threshold=0.5,
                         show_bounding_boxes=True, show_confidence=True, highlight_defects=True):
        """
        Processes a single frame to detect defects.
        Args:
            frame_bgr: The input frame in BGR format.
            target_classes: A list of defect types to detect (e.g., ["Crack", "Rust"]).
            confidence_threshold: Minimum confidence score for a detection to be considered.
                                  This is passed to the underlying YOLO model.
            show_bounding_boxes: Boolean, whether to draw bounding boxes.
            show_confidence: Boolean, whether to display confidence scores.
            highlight_defects: Boolean, whether to apply visual highlights.

        Returns:
            A tuple containing:
            - processed_frame_bgr: The frame with detections drawn on it.
            - detections: A list of dictionaries, each representing a detection.
        """
        if target_classes is None:
            # Default to all known classes if none are specified.
            # This might need adjustment based on AVAILABLE_CLASSES in the Streamlit page.
            target_classes = list(self.class_names.values())

        processed_frame_bgr = frame_bgr.copy()
        detections_summary = []

        if not self.bbox_model: # self.bbox_model is an instance of BoundingBoxModelLoader
            cv2.putText(processed_frame_bgr, "BBox Model Loader not available", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return processed_frame_bgr, detections_summary

        try:
            # Perform prediction using the loader's predict method
            yolo_results_list = self.bbox_model.predict(
                image=frame_bgr,
                conf=confidence_threshold,
                verbose=False,  # Keep verbose off for live feed
                save=False      # No need to save intermediate images from the model's side
            )

            if yolo_results_list:
                yolo_results = yolo_results_list[0]  # Results for the first (and only) image
                model_class_names = yolo_results.names  # Dict {int_idx: str_name} e.g., {0: 'Crack', ...}

                # Ensure all necessary attributes exist
                if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
                    boxes = yolo_results.boxes.xyxy.cpu().numpy()  # [N, 4]
                    confs = yolo_results.boxes.conf.cpu().numpy()  # [N,]
                    cls_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)  # [N,]

                    for i in range(len(boxes)):
                        class_id = cls_ids[i]
                        # Use model's class name; fallback if class_id not in model_class_names
                        class_name = model_class_names.get(class_id, f"UnknownClass_{class_id}")

                        confidence = confs[i]

                        # Filter by target_classes (list of strings like ["Crack", "Rust"])
                        # The confidence filter is primarily handled by `conf` in model.predict,
                        # but target_classes needs to be checked here.
                        if target_classes and class_name not in target_classes:
                            continue

                        bbox_coords = boxes[i]  # [x1, y1, x2, y2]
                        detections_summary.append({
                            "type": class_name,
                            "confidence": float(confidence),
                            "bbox": bbox_coords.tolist(),
                            "class_id": int(class_id)
                        })
                else:
                    print("[DEBUG-LivePredictor] No 'boxes' attribute in YOLO results or it's None.")
            else:
                print("[DEBUG-LivePredictor] YOLO prediction returned empty or None.")

        except Exception as e:
            print(f"[ERROR-LivePredictor] Error during model prediction or processing: {e}")
            cv2.putText(processed_frame_bgr, f"Prediction Error", (30, 60), # Simplified error on frame
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            # Return the original frame (or processed up to error) and empty/partial detections
            return processed_frame_bgr, detections_summary # or frame_bgr.copy() if processed_frame_bgr is compromised

        # Draw detections on the frame using the populated detections_summary
        for det in detections_summary:
            label = det.get("type", "Unknown")
            confidence = det.get("confidence", 0.0)
            bbox = det.get("bbox")  # [x1, y1, x2, y2]

            if bbox:
                x1, y1, x2, y2 = map(int, bbox)

                # Determine color based on defect type
                color = (0, 0, 0) # Default: Black
                if label == "Crack": color = (0, 0, 255)    # Red
                elif label == "Rust": color = (0, 165, 255) # Orange
                elif label == "Tower Structure": color = (0, 255, 0) # Green
                # Add more colors as needed for other classes

                if show_bounding_boxes:
                    cv2.rectangle(processed_frame_bgr, (x1, y1), (x2, y2), color, 2)

                if show_confidence:
                    text = f"{label}: {confidence:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(processed_frame_bgr, (x1, y1 - text_height - 15), (x1 + text_width, y1 - 10), color, -1) # Background for text
                    cv2.putText(processed_frame_bgr, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2) # White text

                if highlight_defects and label != "Tower Structure":  # Example: highlight specific defects
                    try:
                        sub_img = processed_frame_bgr[y1:y2, x1:x2]
                        if sub_img.size > 0: # Ensure sub_img is not empty
                            # Create a colored overlay for highlighting
                            overlay = np.full(sub_img.shape, color, dtype=np.uint8)
                            cv2.addWeighted(sub_img, 0.7, overlay, 0.3, 0, dst=sub_img) # Apply overlay to sub_img
                            processed_frame_bgr[y1:y2, x1:x2] = sub_img # Put it back
                    except Exception as highlight_e:
                        print(f"[WARNING-LivePredictor] Could not apply highlight for {label}: {highlight_e}")


        # --- Placeholder for Segmentation Model Prediction (if applicable) ---
        # if self.seg_model and ("some_segmentable_defect" in target_classes):
        #   masks = self.seg_model.predict(frame_bgr) # Hypothetical call to SegmentationModelLoader
        #   processed_frame_bgr = self.draw_segmentation_masks(processed_frame_bgr, masks)
        #   # Add segmentation detections to detections_summary

        return processed_frame_bgr, detections_summary

    def draw_segmentation_masks(self, image, masks, alpha=0.5):
        """
        Draws segmentation masks on an image. (Helper, if needed)
        """
        # This is a placeholder. Actual implementation depends on mask format.
        # For example, if masks is a list of binary masks:
        for mask in masks: # Assuming mask is a 2D numpy array
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            # Assign a color to the mask, e.g., based on class
            # For demonstration, let's make it semi-transparent blue
            colored_mask[mask > 0] = [255, 0, 0] # Blue where mask is active
            image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        return image

# # Example usage (not part of the class, for testing/understanding):
# if __name__ == '__main__':
#     # This block would not run when imported by Streamlit
#     # You'd need to mock or provide actual BoundingBoxModelLoader and SegmentationModelLoader
#     class MockModelLoader:
#         def load_model(self):
#             print(f"MockModelLoader: Loading mock model...")
#             return "mock_model_instance" # Represents a loaded model

#     bbox_loader = MockModelLoader()
#     seg_loader = MockModelLoader()

#     live_predictor = LiveModelPredictor(bbox_loader, seg_loader)

#     # Create a dummy frame (e.g., black image)
#     dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

#     processed_frame, detections = live_predictor.predict_on_frame(
#         dummy_frame,
#         target_classes=["Crack", "Rust"],
#         confidence_threshold=0.4,
#         show_bounding_boxes=True,
#         show_confidence=True
#     )
#     print("Detections:", detections)
#     # To display with OpenCV (if running locally and have it installed):
#     # cv2.imshow("Processed Frame", processed_frame)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
