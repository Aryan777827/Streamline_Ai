from src.preprocessing.video_processor import VideoProcessor
from src.models.object_detector import ObjectDetector
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    processor = VideoProcessor('data/videos/sample_video.mp4')
    frames = processor.extract_frames(sample_rate=60, max_frames=3)
    print('\nRunning object detection...')
    detector = ObjectDetector(model_size='n')
    all_detections = detector.detect_batch(frames)
    frame = frames[0]
    detections = all_detections[0]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(frame)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"{det['class_name']} {det['confidence']:.2f}", color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    ax.axis('off')
    plt.savefig('data/outputs/detection_result.png', dpi=150, bbox_inches='tight')
    print(f'Detected {len(detections)} objects')
    print('Saved to data/outputs/detection_result.png')
except FileNotFoundError:
    print('\nAdd video to data/videos/sample_video.mp4')
except Exception as e:
    print(f'Error: {e}')
