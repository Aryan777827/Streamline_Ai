from src.inference.video_analyzer import VideoAnalyzer
import json

video_path = "data/videos/sample_video.mp4"

try:
    print("\n🎬 Starting Complete Video Analysis...\n")
    
    # Initialize analyzer
    analyzer = VideoAnalyzer()
    
    # Analyze video
    results = analyzer.analyze_video(video_path)
    
    # Generate and print summary
    summary = analyzer.generate_summary(results)
    print(summary)
    
    # Save detailed results to JSON
    output_path = "data/outputs/complete_analysis.json"
    with open(output_path, 'w') as f:
        # Convert to JSON-serializable format
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    print("\n✅ Analysis Complete!")
    
except FileNotFoundError:
    print("\n❌ Add video to data/videos/sample_video.mp4")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()