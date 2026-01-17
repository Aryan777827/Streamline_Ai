import cv2
import numpy as np
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video loading and frame extraction."""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f'Video not found: {video_path}')
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f'Cannot open video: {video_path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        logger.info(f'Loaded: {self.video_path.name}')
    
    def extract_frames(self, sample_rate: int = 1, max_frames: int = None) -> List[np.ndarray]:
        frames = []
        frame_count = 0
        extracted_count = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_count % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1
                if max_frames and extracted_count >= max_frames:
                    break
            frame_count += 1
        logger.info(f'Extracted {len(frames)} frames')
        return frames
    
    def get_video_info(self) -> dict:
        return {
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    
    def get_frame_at_timestamp(self, timestamp: float) -> np.ndarray:
        """Extract frame at specific timestamp."""
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame at timestamp {timestamp}s")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()