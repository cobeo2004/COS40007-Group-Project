import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.model_loader import BoundingBoxModelLoader, SegmentationModelLoader
from typing import TypeVar, Union, Literal
from interfaces.model_loader_interface import ModelLoaderImageType


class ModelPredictor:
    def __init__(self, bbox_model: BoundingBoxModelLoader, seg_model: SegmentationModelLoader):
        self.bbox_model = bbox_model
        self.seg_model = seg_model

    def run_inference_and_combine(self, test_images_dir, output_dir):
        """Run inference with both models and combine results"""
        # Load models
        bbox_model = self.bbox_model
        seg_model = self.seg_model

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get list of test images
        image_paths = glob.glob(os.path.join(test_images_dir, "*.jpg"))

        if not image_paths:
            print(f"⚠️ No images found in {test_images_dir}")
            return

        print(f"Running inference on {len(image_paths)} images...")

        # Run inference on all test images
        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}...")

            image_name = os.path.basename(image_path)

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Failed to load image: {image_path}")
                continue

            # Run inference with both models
            bbox_results = bbox_model.predict(img, save=False, verbose=False)
            seg_results = seg_model.predict(img, save=False, verbose=False)

            # Create a new image with combined results
            result_img = img.copy()

            # Process bounding box detections (Crack and Rust)
            if bbox_results and len(bbox_results[0].boxes) > 0:
                boxes = bbox_results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf > 0.25:  # Confidence threshold
                        # Class 0 is Crack (red color)
                        if cls == 0:
                            color = (0, 0, 255)  # BGR format: red
                            label = f"Crack {conf:.2f}"
                        # Class 1 is Rust (blue color)
                        else:
                            color = (255, 0, 0)  # BGR format: blue
                            label = f"Rust {conf:.2f}"

                        # Draw bounding box
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

                        # Draw label
                        cv2.putText(result_img, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Process segmentation detections (Tower Structure)
            if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
                masks = seg_results[0].masks

                # Create an overlay for the segmentation
                overlay = result_img.copy()

                for i, mask_tensor in enumerate(masks.data):
                    # Convert mask tensor to numpy array and resize to image dimensions
                    mask = mask_tensor.cpu().numpy()
                    mask = cv2.resize(
                        mask, (result_img.shape[1], result_img.shape[0]))

                    # Apply the mask
                    overlay[mask > 0.5] = overlay[mask > 0.5] * \
                        0.7 + np.array([0, 255, 0]) * 0.3

                # Add the overlay to the result image
                cv2.addWeighted(overlay, 0.7, result_img, 0.3, 0, result_img)

                # Add label for Tower Structure
                cv2.putText(result_img, "Tower Structure", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save the result image
            output_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_path, result_img)

            # Display a sample result every 10 images
            if i % 10 == 0:
                plt.figure(figsize=(12, 8))
                # Convert BGR to RGB for correct display
                plt_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                plt.imshow(plt_img)
                plt.title(f"Detection Results: {image_name}")
                plt.axis('off')
                plt.show()

        print(f"✅ Inference completed. Results saved to: {output_dir}")
