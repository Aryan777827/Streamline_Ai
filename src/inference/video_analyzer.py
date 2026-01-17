from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.scene_detector import SceneDetector
from src.models.object_detector import ObjectDetector
from typing import Dict, List
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Complete video analysis pipeline."""
    
    def __init__(self):
        self.scene_detector = SceneDetector(threshold=27.0)
        self.object_detector = ObjectDetector(model_size='n')
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Perform complete video analysis with scene detection.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting comprehensive analysis of {video_path}")
        
        # Step 1: Get video info
        processor = VideoProcessor(video_path)
        video_info = processor.get_video_info()
        
        # Step 2: Detect scenes
        logger.info("Analyzing scene structure...")
        scene_summary = self.scene_detector.get_scene_summary(video_path)
        scenes = scene_summary.get('scenes', [])
        
        # Step 3: Get key frame timestamps
        key_timestamps = self.scene_detector.get_key_frames_timestamps(video_path)
        
        # Step 4: Extract key frames
        logger.info("Extracting key frames...")
        key_frames = []
        for ts in key_timestamps:
            frame = processor.get_frame_at_timestamp(ts)
            key_frames.append(frame)
        
        # Step 5: Detect objects in key frames
        logger.info("Running object detection on key frames...")
        all_detections = self.object_detector.detect_batch(key_frames)
        
        # Step 6: Analyze detected objects
        object_summary = self._summarize_objects(all_detections)
        
        # Compile results
        results = {
            'video_info': video_info,
            'scene_analysis': scene_summary,
            'num_key_frames': len(key_frames),
            'key_timestamps': key_timestamps,
            'object_summary': object_summary,
            'scene_details': []
        }
        
        # Add per-scene details
        for i, (detections, timestamp) in enumerate(zip(all_detections, key_timestamps)):
            scene_info = {
                'scene_id': i + 1,
                'timestamp': timestamp,
                'objects_detected': len(detections),
                'objects': [det['class_name'] for det in detections]
            }
            results['scene_details'].append(scene_info)
        
        logger.info("Analysis complete")
        return results
    
    def _summarize_objects(self, all_detections: List[List[Dict]]) -> Dict:
        """Summarize objects across all frames."""
        all_objects = []
        for detections in all_detections:
            for det in detections:
                all_objects.append(det['class_name'])
        
        object_counts = Counter(all_objects)
        total_objects = len(all_objects)
        
        return {
            'total_objects_detected': total_objects,
            'unique_objects': len(object_counts),
            'top_objects': object_counts.most_common(5)
        }
    
    def generate_summary(self, analysis_results: Dict) -> str:
        """Generate human-readable summary."""
        info = analysis_results['video_info']
        scene_analysis = analysis_results['scene_analysis']
        obj_summary = analysis_results['object_summary']
        
        summary = f"""
VIDEO ANALYSIS SUMMARY
{'='*60}

Video Details:
- Duration: {info['duration']:.2f} seconds
- Resolution: {info['width']}x{info['height']}
- FPS: {info['fps']:.2f}
- Total Frames: {info['total_frames']}

Scene Analysis:
- Total Scenes Detected: {scene_analysis['total_scenes']}
- Average Scene Length: {scene_analysis['avg_scene_length']:.2f}s
- Shortest Scene: {scene_analysis['shortest_scene']:.2f}s
- Longest Scene: {scene_analysis['longest_scene']:.2f}s

Object Detection (from {analysis_results['num_key_frames']} key frames):
- Total Objects Detected: {obj_summary['total_objects_detected']}
- Unique Object Types: {obj_summary['unique_objects']}
- Most Common Objects:
"""
        
        for obj, count in obj_summary['top_objects']:
            summary += f"  • {obj}: {count} occurrences\n"
        
        summary += "\nScene Breakdown:\n"
        for scene in analysis_results['scene_details'][:5]:  # Show first 5
            summary += f"  Scene {scene['scene_id']} ({scene['timestamp']:.1f}s): "
            summary += f"{scene['objects_detected']} objects - {', '.join(list(set(scene['objects']))[:3])}\n" 
        
        return summary