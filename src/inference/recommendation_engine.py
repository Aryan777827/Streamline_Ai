from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generate video recommendations based on content similarity."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.video_database = []
        self.feature_vectors = None
        logger.info("Recommendation engine initialized")
    
    def extract_features(self, analysis_results: Dict) -> str:
        """Extract text features from video analysis."""
        features = []
        
        obj_summary = analysis_results.get('object_summary', {})
        top_objects = obj_summary.get('top_objects', [])
        for obj, count in top_objects:
            features.extend([obj] * min(count, 5))
        
        scene_count = analysis_results.get('scene_analysis', {}).get('total_scenes', 0)
        if scene_count > 0:
            features.append(f"scenes_{scene_count}")
        
        duration = analysis_results.get('video_info', {}).get('duration', 0)
        if duration < 10:
            features.append("short_video")
        elif duration < 30:
            features.append("medium_video")
        else:
            features.append("long_video")
        
        return ' '.join(features)
    
    def add_video(self, video_id: str, analysis_results: Dict):
        """Add a video to the recommendation database."""
        features = self.extract_features(analysis_results)
        self.video_database.append({
            'id': video_id,
            'features': features,
            'analysis': analysis_results
        })
        logger.info(f"Added video '{video_id}' to database")
    
    def build_index(self):
        """Build the feature index for all videos."""
        if not self.video_database:
            logger.warning("No videos in database")
            return
        
        feature_strings = [video['features'] for video in self.video_database]
        self.feature_vectors = self.vectorizer.fit_transform(feature_strings)
        logger.info(f"Built index for {len(self.video_database)} videos")
    
    def get_recommendations(self, video_id: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get recommended videos similar to the given video."""
        if self.feature_vectors is None:
            logger.error("Index not built")
            return []
        
        video_idx = None
        for idx, video in enumerate(self.video_database):
            if video['id'] == video_id:
                video_idx = idx
                break
        
        if video_idx is None:
            logger.error(f"Video '{video_id}' not found")
            return []
        
        video_vector = self.feature_vectors[video_idx]
        similarities = cosine_similarity(video_vector, self.feature_vectors).flatten()
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        
        recommendations = []
        for idx in similar_indices:
            video = self.video_database[idx]
            score = similarities[idx]
            recommendations.append((video['id'], score))
        
        return recommendations
    
    def get_video_info(self, video_id: str) -> Dict:
        """Get stored analysis for a video."""
        for video in self.video_database:
            if video['id'] == video_id:
                return video['analysis']
        return None