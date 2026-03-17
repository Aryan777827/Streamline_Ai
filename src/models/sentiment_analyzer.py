from transformers import pipeline
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment of text."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        logger.info("Loading sentiment analysis model")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        logger.info("Sentiment model loaded")
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment result with label and score
        """
        if not text or len(text.strip()) == 0:
            return {'label': 'NEUTRAL', 'score': 0.0}
        
        if len(text) > 512:
            text = text[:512]
        
        result = self.sentiment_pipeline(text)[0]
        return result
    
    def analyze_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment of multiple text segments.
        
        Args:
            segments: List of segments with 'text' field
            
        Returns:
            Segments with sentiment added
        """
        logger.info(f"Analyzing sentiment for {len(segments)} segments")
        
        for segment in segments:
            sentiment = self.analyze_text(segment['text'])
            segment['sentiment'] = sentiment['label']
            segment['sentiment_score'] = sentiment['score']
        
        return segments
    
    def get_overall_sentiment(self, segments: List[Dict]) -> Dict:
        """
        Calculate overall sentiment from segments.
        
        Args:
            segments: List of segments with sentiment
            
        Returns:
            Overall sentiment summary
        """
        if not segments:
            return {'label': 'NEUTRAL', 'confidence': 0.0, 'distribution': {}}
        
        positive_count = sum(1 for s in segments if s.get('sentiment') == 'POSITIVE')
        negative_count = sum(1 for s in segments if s.get('sentiment') == 'NEGATIVE')
        
        total = len(segments)
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        
        if positive_ratio > negative_ratio:
            label = 'POSITIVE'
            confidence = positive_ratio
        elif negative_ratio > positive_ratio:
            label = 'NEGATIVE'
            confidence = negative_ratio
        else:
            label = 'NEUTRAL'
            confidence = 0.5
        
        return {
            'label': label,
            'confidence': confidence,
            'distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': total - positive_count - negative_count
            }
        }