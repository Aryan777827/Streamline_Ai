from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.scene_detector import SceneDetector
from src.preprocessing.audio_processor import AudioProcessor
from src.models.object_detector import ObjectDetector
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.inference.recommendation_engine import RecommendationEngine
from typing import Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompletePipeline:
    """Complete end-to-end video analysis pipeline."""
    
    def __init__(self, enable_audio: bool = True, enable_sentiment: bool = True):
        """
        Initialize complete pipeline.
        
        Args:
            enable_audio: Enable audio transcription
            enable_sentiment: Enable sentiment analysis
        """
        logger.info("Initializing complete pipeline...")
        
        # Core components
        self.video_processor = None
        self.scene_detector = SceneDetector(threshold=15.0)
        self.object_detector = ObjectDetector(model_size='n')
        
        # Optional components
        self.enable_audio = enable_audio
        self.enable_sentiment = enable_sentiment
        
        if enable_audio:
            self.audio_processor = AudioProcessor(model_size='base')
        
        if enable_sentiment:
            self.sentiment_analyzer = SentimentAnalyzer()
        
        logger.info("Pipeline initialized successfully")
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Run complete analysis on a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting complete analysis: {video_path}")
        results = {}
        
        # 1. Video Info
        logger.info("Step 1/5: Extracting video information...")
        self.video_processor = VideoProcessor(video_path)
        results['video_info'] = self.video_processor.get_video_info()
        
        # 2. Scene Detection
        logger.info("Step 2/5: Detecting scenes...")
        scene_summary = self.scene_detector.get_scene_summary(video_path)
        key_timestamps = self.scene_detector.get_key_frames_timestamps(video_path)
        results['scene_analysis'] = scene_summary
        results['key_timestamps'] = key_timestamps
        
        # 3. Object Detection
        logger.info("Step 3/5: Detecting objects in key frames...")
        key_frames = []
        for ts in key_timestamps:
            frame = self.video_processor.get_frame_at_timestamp(ts)
            key_frames.append(frame)
        
        all_detections = self.object_detector.detect_batch(key_frames)
        results['object_detections'] = all_detections
        results['object_summary'] = self._summarize_objects(all_detections)
        
        # 4. Audio Transcription (if enabled)
        if self.enable_audio:
            logger.info("Step 4/5: Transcribing audio...")
            try:
                audio_analysis = self.audio_processor.transcribe_video(video_path)
                results['audio_analysis'] = audio_analysis
                
                # 5. Sentiment Analysis (if enabled and audio available)
                if self.enable_sentiment and audio_analysis.get('segments'):
                    logger.info("Step 5/5: Analyzing sentiment...")
                    segments_with_sentiment = self.sentiment_analyzer.analyze_segments(
                        audio_analysis['segments']
                    )
                    overall_sentiment = self.sentiment_analyzer.get_overall_sentiment(
                        segments_with_sentiment
                    )
                    results['audio_analysis']['segments'] = segments_with_sentiment
                    results['audio_analysis']['overall_sentiment'] = overall_sentiment
                else:
                    logger.info("Step 5/5: Skipping sentiment (no audio segments)")
            except Exception as e:
                import traceback; logger.error(f"Audio analysis failed: {traceback.format_exc()}")
                results['audio_analysis'] = None
        else:
            logger.info("Step 4-5/5: Skipping audio analysis (disabled)")
            results['audio_analysis'] = None
        
        logger.info("Complete analysis finished!")
        return results
    
    def _summarize_objects(self, all_detections):
        """Summarize detected objects."""
        from collections import Counter
        all_objects = []
        for detections in all_detections:
            for det in detections:
                all_objects.append(det['class_name'])
        
        object_counts = Counter(all_objects)
        return {
            'total_objects_detected': len(all_objects),
            'unique_objects': len(object_counts),
            'top_objects': object_counts.most_common(5)
        }
    
    def generate_report(self, analysis_results: Dict) -> str:
        """Generate human-readable report."""
        info = analysis_results['video_info']
        scene = analysis_results['scene_analysis']
        obj = analysis_results['object_summary']
        
        report = f"""
{'='*70}
COMPLETE VIDEO ANALYSIS REPORT
{'='*70}

VIDEO INFORMATION
-----------------
Duration: {info['duration']:.2f}s
Resolution: {info['width']}x{info['height']}
FPS: {info['fps']:.2f}
Total Frames: {info['total_frames']}

SCENE ANALYSIS
--------------
Total Scenes: {scene['total_scenes']}
Average Scene Length: {scene['avg_scene_length']:.2f}s
Shortest Scene: {scene['shortest_scene']:.2f}s
Longest Scene: {scene['longest_scene']:.2f}s

OBJECT DETECTION
----------------
Total Objects: {obj['total_objects_detected']}
Unique Types: {obj['unique_objects']}
Top Objects:
"""
        for obj_name, count in obj['top_objects']:
            report += f"  • {obj_name}: {count}\n"
        
        # Add audio/sentiment if available
        audio = analysis_results.get('audio_analysis')
        if audio:
            report += f"\nAUDIO TRANSCRIPTION\n-------------------\n"
            report += f"Language: {audio.get('language', 'N/A')}\n"
            report += f"Segments: {len(audio.get('segments', []))}\n"
            
            if audio.get('transcription'):
                report += f"\nTranscript Preview:\n{audio['transcription'][:200]}...\n"
            
            if audio.get('overall_sentiment'):
                sentiment = audio['overall_sentiment']
                report += f"\nSENTIMENT ANALYSIS\n------------------\n"
                report += f"Overall: {sentiment['label']} ({sentiment['confidence']:.1%})\n"
                report += f"Distribution: {sentiment['distribution']}\n"
        
        report += f"\n{'='*70}\n"
        return report
    
    def save_results(self, analysis_results: Dict, output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json_results = json.loads(json.dumps(analysis_results, default=str))
            json.dump(json_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
