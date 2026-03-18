from src.inference.recommendation_engine import RecommendationEngine

# Sample video analysis results
sample_videos = {
    "car_chase_1": {
        "video_info": {"duration": 15.5, "fps": 30},
        "object_summary": {
            "total_objects_detected": 25,
            "top_objects": [("car", 15), ("person", 5), ("traffic light", 5)]
        },
        "scene_analysis": {"total_scenes": 5}
    },
    "car_chase_2": {
        "video_info": {"duration": 18.2, "fps": 30},
        "object_summary": {
            "total_objects_detected": 30,
            "top_objects": [("car", 18), ("truck", 7), ("person", 5)]
        },
        "scene_analysis": {"total_scenes": 6}
    },
    "nature_doc_1": {
        "video_info": {"duration": 45.0, "fps": 24},
        "object_summary": {
            "total_objects_detected": 20,
            "top_objects": [("bird", 12), ("tree", 8)]
        },
        "scene_analysis": {"total_scenes": 3}
    },
    "nature_doc_2": {
        "video_info": {"duration": 50.0, "fps": 24},
        "object_summary": {
            "total_objects_detected": 18,
            "top_objects": [("bird", 10), ("tree", 5), ("bench", 3)]
        },
        "scene_analysis": {"total_scenes": 4}
    },
    "cooking_show": {
        "video_info": {"duration": 8.0, "fps": 30},
        "object_summary": {
            "total_objects_detected": 15,
            "top_objects": [("bowl", 5), ("spoon", 4), ("bottle", 3), ("cup", 3)]
        },
        "scene_analysis": {"total_scenes": 2}
    },
    "sports_game": {
        "video_info": {"duration": 25.0, "fps": 60},
        "object_summary": {
            "total_objects_detected": 35,
            "top_objects": [("person", 22), ("sports ball", 8), ("bench", 5)]
        },
        "scene_analysis": {"total_scenes": 8}
    }
}

print("\n" + "="*60)
print("RECOMMENDATION ENGINE TEST")
print("="*60)

# Initialize engine
engine = RecommendationEngine()

# Add all videos to database
print("\nAdding videos to database...")
for video_id, analysis in sample_videos.items():
    engine.add_video(video_id, analysis)

# Build the recommendation index
print("\nBuilding recommendation index...")
engine.build_index()

# Test recommendations for different videos
test_videos = ["car_chase_1", "nature_doc_1", "cooking_show"]

for test_video in test_videos:
    print(f"\n{'='*60}")
    print(f"Recommendations for: {test_video}")
    print("="*60)
    
    # Get video info
    info = engine.get_video_info(test_video)
    print(f"\nVideo Details:")
    print(f"  Duration: {info['video_info']['duration']:.1f}s")
    print(f"  Scenes: {info['scene_analysis']['total_scenes']}")
    top_obj_str = ', '.join([f'{obj}({count})' for obj, count in info['object_summary']['top_objects'][:3]])
    print(f"  Top Objects: {top_obj_str}")
    
    # Get recommendations
    recommendations = engine.get_recommendations(test_video, top_n=3)
    
    print(f"\nTop 3 Similar Videos:")
    for i, (rec_id, score) in enumerate(recommendations, 1):
        rec_info = engine.get_video_info(rec_id)
        top_objects = ', '.join([obj for obj, _ in rec_info['object_summary']['top_objects'][:2]])
        print(f"\n{i}. {rec_id} (similarity: {score:.2%})")
        print(f"   Objects: {top_objects}")
        print(f"   Duration: {rec_info['video_info']['duration']:.1f}s")

print("\n" + "="*60)
print("✅ Recommendation Engine Test Complete!")
print("="*60)