# ==============================================================================
# BACKEND - AI & COMPUTER VISION LOGIC (v5 - Final)
#
# This version is designed to work with the latest versions of all libraries
# to ensure full compatibility with modern PyTorch and Apple Silicon.
# ==============================================================================

import cv2
import numpy as np
import time
import json
from datetime import datetime
import torch
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. AI OBJECT DETECTOR (Corrected for latest libraries)
# ==============================================================================

class AIObjectDetector:
    """AI-powered object detection using YOLOv8."""

    def __init__(self, model_name='yolov8m.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_name)
        
        if self.model is None:
            raise RuntimeError(
                f"YOLOv8 model file '{model_name}' failed to load. "
                "Ensure the file exists in the same directory as backend.py, is not corrupted, "
                "and that you have run 'pip install --upgrade -r requirements.txt' in your virtual environment."
            )
        
        self.class_names = self.model.names

    def _load_model(self, model_name):
        """Loads a pre-trained YOLOv8 model from a local file path."""
        try:
            # The latest ultralytics library correctly prioritizes the local file.
            # Simply passing the name is the most robust method.
            print(f"Attempting to load model '{model_name}' from local path...")
            model = YOLO(model_name)
            model.to(self.device)
            print("AI Model (YOLOv8) Loaded Successfully.")
            return model
        except Exception as e:
            print(f"FATAL: YOLOv8 model loading failed: {e}")
            return None

    def detect_objects(self, image, confidence_threshold=0.5):
        if self.model is None: raise Exception("AI model (YOLOv8) is not loaded.")
        detections = []
        try:
            results = self.model(image, conf=confidence_threshold, verbose=False)
            for result in results:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    detections.append({
                        'id': i, 'class_name': self.model.names[class_id], 'class_id': class_id,
                        'confidence': float(box.conf[0]), 'bbox': list(map(int, box.xyxy[0])),
                        'center': [int((box.xyxy[0][0] + box.xyxy[0][2]) / 2), int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)],
                        'width': int(box.xyxy[0][2] - box.xyxy[0][0]), 'height': int(box.xyxy[0][3] - box.xyxy[0][1])
                    })
            return detections
        except Exception as e:
            print(f"YOLOv8 detection error: {e}"); return []

# ==============================================================================
# 2. CAMERA CONTROLLER & 3. 3D MEASUREMENT ENGINE (No Changes)
# The rest of this file is unchanged and correct.
# ==============================================================================

class AICameraController:
    def __init__(self, camera_url): self.camera_url = camera_url
    def capture_single_image(self):
        try:
            source = int(self.camera_url) if self.camera_url.isdigit() else self.camera_url
            cap = cv2.VideoCapture(source)
            if not cap.isOpened(): raise ConnectionError(f"Could not open camera stream at {self.camera_url}")
            for _ in range(5): cap.read(); time.sleep(0.05)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None: raise IOError(f"Failed to capture frame from {self.camera_url}")
            return frame
        except (ConnectionError, IOError) as e: raise e
        except Exception as e: raise Exception(f"An unexpected error: {e}")

class Auto3DMeasurement:
    def __init__(self, detector): self.detector = detector; self.reference_objects = {'cell phone': 15.0, 'book': 21.0, 'bottle': 8.0, 'cup': 10.0, 'laptop': 35.0, 'remote': 18.0, 'keyboard': 45.0, 'mouse': 12.0, 'tv': 120.0, 'chair': 50.0}
    def _match_objects(self, left_detections, right_detections):
        matched_objects = []
        for left_obj in left_detections:
            best_match, best_score = None, 0
            for right_obj in right_detections:
                if left_obj['class_name'] == right_obj['class_name']:
                    y_diff = abs(left_obj['center'][1] - right_obj['center'][1]); max_height = max(left_obj['height'], right_obj['height'], 1); height_sim = min(left_obj['height'], right_obj['height']) / max_height; position_score = 1.0 - (y_diff / 720.0); score = position_score * height_sim
                    if score > best_score and score > 0.3: best_score, best_match = score, right_obj
            if best_match: matched_objects.append({'class_name': left_obj['class_name'], 'left_bbox': left_obj['bbox'], 'right_bbox': best_match['bbox'], 'left_center': left_obj['center'], 'right_center': best_match['center'], 'confidence': (left_obj['confidence'] + best_match['confidence']) / 2})
        return matched_objects
    def _calculate_3d_measurements(self, matched_objects):
        measurements = []; baseline_cm, focal_length_px = 15.0, 1000.0
        for obj in matched_objects:
            disparity_px = abs(obj['left_center'][0] - obj['right_center'][0])
            if disparity_px < 5: continue
            depth_cm = max(20.0, min(500.0, (baseline_cm * focal_length_px) / disparity_px)); pixel_width = obj['left_bbox'][2] - obj['left_bbox'][0]; real_width_cm = (pixel_width * depth_cm) / focal_length_px; pixel_height = obj['left_bbox'][3] - obj['left_bbox'][1]; real_height_cm = (pixel_height * depth_cm) / focal_length_px
            measurements.append({'Object': obj['class_name'].title(), 'Confidence': f"{obj['confidence']:.2f}", 'Depth (cm)': f"{depth_cm:.1f}", 'Width (cm)': f"{real_width_cm:.1f}", 'Height (cm)': f"{real_height_cm:.1f}", '2D Position': obj['left_center'], 'Disparity (px)': disparity_px})
        return measurements
    def run_full_measurement_process(self, left_image, right_image):
        print("Detecting objects..."); left_detections = self.detector.detect_objects(left_image, 0.5); right_detections = self.detector.detect_objects(right_image, 0.5); print(f"Found {len(left_detections)} in left, {len(right_detections)} in right.")
        matched_objects = self._match_objects(left_detections, right_detections)
        if not matched_objects: return None, None
        print(f"Matched {len(matched_objects)} objects."); measurements = self._calculate_3d_measurements(matched_objects); result_image = left_image.copy()
        for m in measurements: pos = m['2D Position']; color = (0, 255, 0); cv2.circle(result_image, tuple(pos), 10, color, -1); cv2.putText(result_image, f"{m['Object']} @ {m['Depth (cm)']} cm", (pos[0] + 15, pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return measurements, result_image
