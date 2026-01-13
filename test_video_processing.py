from src.preprocessing.video_processor import VideoProcessor
import matplotlib.pyplot as plt

try:
    processor = VideoProcessor('data/videos/sample_video.mp4')
    info = processor.get_video_info()
    print('\nVideo Info:', info)
    frames = processor.extract_frames(sample_rate=30, max_frames=5)
    print(f'Extracted {len(frames)} frames')
    plt.imshow(frames[0])
    plt.axis('off')
    plt.savefig('data/outputs/first_frame.png')
    print('Saved to data/outputs/first_frame.png')
except FileNotFoundError:
    print('\nAdd video to data/videos/sample_video.mp4')
except Exception as e:
    print(f'Error: {e}')
