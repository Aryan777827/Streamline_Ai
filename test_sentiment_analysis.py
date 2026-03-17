from src.models.sentiment_analyzer import SentimentAnalyzer

# Sample texts with different sentiments
test_texts = [
    "This is absolutely amazing! I love it so much!",
    "This is terrible and disappointing.",
    "The weather is okay today.",
    "I'm really excited about this new opportunity!",
    "I'm feeling sad and frustrated.",
    "This product works as expected."
]

try:
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS TEST")
    print("="*60)
    
    # Initialize analyzer
    print("\nInitializing sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    
    # Analyze each text
    print("\nAnalyzing sample texts:\n")
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_text(text)
        print(f"{i}. Text: \"{text}\"")
        print(f"   Sentiment: {result['label']} (confidence: {result['score']:.2%})")
        print()
    
    # Test with segments
    segments = [
        {'text': 'This is great!', 'start': 0.0, 'end': 2.0},
        {'text': 'I hate this part.', 'start': 2.0, 'end': 4.0},
        {'text': 'Overall it was fine.', 'start': 4.0, 'end': 6.0}
    ]
    
    print("\nAnalyzing segments:")
    segments_with_sentiment = analyzer.analyze_segments(segments)
    for seg in segments_with_sentiment:
        print(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")
        print(f"  → {seg['sentiment']} ({seg['sentiment_score']:.2%})")
    
    # Get overall sentiment
    overall = analyzer.get_overall_sentiment(segments_with_sentiment)
    print(f"\nOverall Sentiment: {overall['label']} ({overall['confidence']:.2%})")
    print(f"Distribution: {overall['distribution']}")
    
    print("\n✅ Sentiment Analysis Complete!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()