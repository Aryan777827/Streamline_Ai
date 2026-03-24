from moviepy import VideoFileClip
from pathlib import Path
import whisper
import torch
import subprocess
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_MINUTES = 10

class AudioProcessor:
    def __init__(self, model_size='base'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Loading Whisper model: {model_size} on {self.device}')
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info('Whisper model loaded')

    def extract_audio(self, video_path, output_path=None):
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f'data/temp/{video_name}_audio.wav'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Extracting audio from {Path(video_path).name}')
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            raise ValueError('Video has no audio track!')
        video.audio.write_audiofile(output_path, logger=None)
        video.close()
        logger.info(f'Audio saved to {output_path}')
        return output_path

    def get_duration(self, audio_path):
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True, text=True)
        return float(result.stdout.strip())

    def extract_chunk(self, audio_path, start, duration, chunk_path):
        subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', str(start), '-t', str(duration),
            '-ar', '16000', '-ac', '1', chunk_path
        ], capture_output=True)

    def transcribe(self, audio_path):
        duration = self.get_duration(audio_path)
        chunk_sec = CHUNK_MINUTES * 60
        n_chunks = math.ceil(duration / chunk_sec)
        logger.info(f'Duration: {duration/60:.1f} min, splitting into {n_chunks} chunks')
        all_segments = []
        full_text = []
        language = None
        for i in range(n_chunks):
            start = i * chunk_sec
            chunk_dur = min(chunk_sec, duration - start)
            chunk_path = f'{audio_path}_chunk_{i}.wav'
            self.extract_chunk(audio_path, start, chunk_dur, chunk_path)
            logger.info(f'Transcribing chunk {i+1}/{n_chunks}')
            result = self.model.transcribe(chunk_path, fp16=(self.device=='cuda'))
            Path(chunk_path).unlink(missing_ok=True)
            if language is None:
                language = result.get('language', 'unknown')
            full_text.append(result['text'].strip())
            for seg in result['segments']:
                all_segments.append({
                    'start': round(seg['start'] + start, 2),
                    'end': round(seg['end'] + start, 2),
                    'text': seg['text'].strip()
                })
        logger.info(f'Transcription complete: {len(all_segments)} segments')
        return {'transcription': ' '.join(full_text), 'text': ' '.join(full_text), 'language': language, 'segments': all_segments}

    def transcribe_video(self, video_path):
        audio_path = self.extract_audio(video_path)
        try:
            result = self.transcribe(audio_path)
        finally:
            Path(audio_path).unlink(missing_ok=True)
        return result
