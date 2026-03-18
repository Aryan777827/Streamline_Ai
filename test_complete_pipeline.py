from src.inference.complete_pipeline import CompletePipeline
import sys

video_path = "data/videos/sample_video.mp4"

print("\n" + "="*70)
print("COMPLETE PIPELINE TEST")
print("="*70)
print("\nThis will run ALL analysis features on your video:")
print("  1. Video information extraction")
print("  2. Scene detection")
print("  3. Object detection")
print("  4. Audio transcription (if video has audio)")
print("  5. Sentiment analysis (if audio available)")
print("\n" + "="*70 + "\n")

try:
    # Initialize pipeline
    # Set enable_audio=False if your video has no audio
    pipeline = CompletePipeline(enable_audio=True, enable_sentiment=True)
    
    # Run complete analysis
    print("Running complete analysis...\n")
    results = pipeline.analyze_video(video_path)
    
    # Generate report
    report = pipeline.generate_report(results)
    print(report)
    
    # Save results
    output_path = "data/outputs/complete_pipeline_results.json"
    pipeline.save_results(results, output_path)
    
    print(f"\n💾 Full results saved to: {output_path}")
    print("\n✅ Complete Pipeline Test Successful!")

except FileNotFoundError:
    print("\n❌ Error: Video file not found at data/videos/sample_video.mp4")
    print("   Make sure you have a video in that location.")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()