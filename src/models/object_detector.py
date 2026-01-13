from ultralytics import YOLO
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_size: str = 'n'):
        model_name = f'yolov8{model_size}.pt'
        logger.info(f'Loading YOLO model: {model_name}')
        self.model = YOLO(model_name)
        logger.info('Model loaded')
    
    def detect_objects(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        results = self.model(frame, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold:
                    detection = {
                        'class_id': int(box.cls[0]),
                        'class_name': result.names[int(box.cls[0])],
                        'confidence': confidence,
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    }
                    detections.append(detection)
        return detections
    
    def detect_batch(self, frames: List[np.ndarray], confidence_threshold: float = 0.5) -> List[List[Dict]]:
        all_detections = []
        for i, frame in enumerate(frames):
            detections = self.detect_objects(frame, confidence_threshold)
            all_detections.append(detections)
            logger.info(f'Frame {i+1}/{len(frames)}: {len(detections)} objects')
        return all_detections
