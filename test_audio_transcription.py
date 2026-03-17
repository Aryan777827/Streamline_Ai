from src.preprocessing.audio_processor import AudioProcessor

video_path = "data/videos/sample_video.mp4"

try:
    print("\n" + "="*60)
    print("AUDIO TRANSCRIPTION TEST")
    print("="*60)
    
    # Initialize audio processor with base model
    processor = AudioProcessor(model_size="base")
    
    # Transcribe video
    print("\nTranscribing video...")
    result = processor.transcribe_video(video_path)
    
    # Display results
    print(f"\nDetected Language: {result['language']}")
    print(f"\nFull Transcription:")
    print("-" * 60)
    print(result['text'])
    print("-" * 60)
    
    print(f"\nSegments ({len(result['segments'])} total):")
    for i, segment in enumerate(result['segments'][:5], 1):  # Show first 5
        print(f"\n[{segment['start']:.2f}s - {segment['end']:.2f}s]")
        print(f"  {segment['text']}")
    
    if len(result['segments']) > 5:
        print(f"\n... and {len(result['segments']) - 5} more segments")
    
    print("\n✅ Transcription Complete!")

except FileNotFoundError:
    print("\n❌ Video file not found at data/videos/sample_video.mp4")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()