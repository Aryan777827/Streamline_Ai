from scenedetect import detect, ContentDetector, AdaptiveDetector
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneDetector:
    """Detect scene changes in videos."""
    
    def __init__(self, threshold: float = 27.0):
        """
        Initialize scene detector.
        
        Args:
            threshold: Sensitivity for scene detection (lower = more sensitive)
        """
        self.threshold = threshold
    
    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect scenes in video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        logger.info(f"Detecting scenes in {Path(video_path).name}")
        
        # Detect scenes using ContentDetector
        scene_list = detect(video_path, ContentDetector(threshold=self.threshold))
        
        # Convert to timestamps
        scenes = []
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scenes.append((start_time, end_time))
            logger.info(f"Scene {i+1}: {start_time:.2f}s - {end_time:.2f}s ({end_time-start_time:.2f}s)")
        
        logger.info(f"Detected {len(scenes)} scenes")
        return scenes
    
    def get_key_frames_timestamps(self, video_path: str) -> List[float]:
        """
        Get timestamps of key frames (middle of each scene).
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of timestamps in seconds
        """
        scenes = self.detect_scenes(video_path)
        
        # Get middle frame of each scene as key frame
        key_timestamps = []
        for start, end in scenes:
            mid_time = (start + end) / 2
            key_timestamps.append(mid_time)
        
        logger.info(f"Extracted {len(key_timestamps)} key frame timestamps")
        return key_timestamps
    
    def get_scene_summary(self, video_path: str) -> dict:
        """
        Get comprehensive scene analysis.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with scene statistics
        """
        scenes = self.detect_scenes(video_path)
        
        if not scenes:
            return {
                'total_scenes': 0,
                'avg_scene_length': 0,
                'shortest_scene': 0,
                'longest_scene': 0
            }
        
        scene_lengths = [end - start for start, end in scenes]
        
        return {
            'total_scenes': len(scenes),
            'avg_scene_length': sum(scene_lengths) / len(scene_lengths),
            'shortest_scene': min(scene_lengths),
            'longest_scene': max(scene_lengths),
            'scenes': scenes
        }