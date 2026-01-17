from src.preprocessing.scene_detector import SceneDetector
from src.preprocessing.video_processor import VideoProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

video_path = "data/videos/sample_video.mp4"

try:
    # Test scene detection
    print("\n" + "="*60)
    print("SCENE DETECTION TEST")
    print("="*60)
    
    detector = SceneDetector(threshold=27.0)
    
    # Get scene summary
    summary = detector.get_scene_summary(video_path)
    
    print(f"\nTotal Scenes: {summary['total_scenes']}")
    print(f"Average Scene Length: {summary['avg_scene_length']:.2f}s")
    print(f"Shortest Scene: {summary['shortest_scene']:.2f}s")
    print(f"Longest Scene: {summary['longest_scene']:.2f}s")
    
    # Get key frames
    key_timestamps = detector.get_key_frames_timestamps(video_path)
    print(f"\nKey Frame Timestamps: {[f'{t:.2f}s' for t in key_timestamps]}")
    
    # Extract and visualize key frames
    processor = VideoProcessor(video_path)
    
    num_frames = min(4, len(key_timestamps))  # Show up to 4 frames
    fig, axes = plt.subplots(1, num_frames, figsize=(16, 4))
    
    if num_frames == 1:
        axes = [axes]
    
    for i in range(num_frames):
        frame = processor.get_frame_at_timestamp(key_timestamps[i])
        axes[i].imshow(frame)
        axes[i].set_title(f"Scene {i+1}\n@{key_timestamps[i]:.2f}s")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/outputs/key_frames.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved key frames visualization to data/outputs/key_frames.png")
    
except FileNotFoundError:
    print("\n❌ Add video to data/videos/sample_video.mp4")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()