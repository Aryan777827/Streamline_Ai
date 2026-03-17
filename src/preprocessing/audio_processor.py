from moviepy import VideoFileClip
from pathlib import Path
import whisper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Extract and transcribe audio from videos."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize audio processor.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model loaded")
    
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        """
        Extract audio from video.
        
        Args:
            video_path: Path to video file
            output_path: Path to save audio file
            
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"data/temp/{video_name}_audio.wav"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting audio from {Path(video_path).name}")
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            video.close()
            raise ValueError("Video has no audio track!")
        
        video.audio.write_audiofile(output_path, logger=None)
        video.close()
        
        logger.info(f"Audio saved to {output_path}")
        return output_path
    
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        logger.info(f"Transcribing {Path(audio_path).name}")
        result = self.model.transcribe(audio_path)
        
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        logger.info(f"Transcription complete: {len(segments)} segments")
        
        return {
            'text': result['text'],
            'language': result['language'],
            'segments': segments
        }
    
    def transcribe_video(self, video_path: str) -> dict:
        """
        Extract audio and transcribe in one step.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Transcription results
        """
        audio_path = self.extract_audio(video_path)
        transcription = self.transcribe(audio_path)
        
        # Cleanup temp audio file
        Path(audio_path).unlink(missing_ok=True)
        
        return transcription