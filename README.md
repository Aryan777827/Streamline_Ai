# StreamlineAI - AI-Powered Video Analysis Platform

A full-stack AI video intelligence platform built as a 2028 graduation portfolio project.

## What it does

Upload any video and StreamlineAI automatically runs a complete AI pipeline and displays results in a web dashboard.

## Features

### Video Processing
- Frame extraction at configurable sample rates
- Video metadata extraction (FPS, resolution, duration, frame count)
- Frame-level timestamp access

### Scene Detection
- Automatic scene cut detection
- Scene boundary timestamps
- Scene statistics (total scenes, shortest, longest, average length)
- Key frame extraction per scene

### Object Detection
- YOLOv8n model (COCO-80 classes)
- Per-frame bounding boxes with confidence scores
- Batch detection across all key frames
- Object frequency summary across entire video

### Audio Transcription
- OpenAI Whisper integration
- Full transcript with word-level timestamps
- Language detection
- Segment-level transcription

### Sentiment Analysis
- Per-segment sentiment scoring
- Overall video sentiment (positive/negative/neutral)
- Confidence scores per segment
- Sentiment distribution across transcript

### Recommendation Engine
- Content-based video analysis recommendations
- Insight generation from combined pipeline results

### Web Dashboard
- Drag and drop video upload
- Live processing progress bar with pipeline step indicators
- Scene timeline with per-scene object inspection
- Object detection frequency chart
- Audio transcript viewer
- Raw JSON export and download
- FastAPI backend with REST endpoints

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn |
| Computer Vision | OpenCV, YOLOv8 (Ultralytics) |
| Speech | OpenAI Whisper |
| Deep Learning | PyTorch |
| NLP | Transformers (HuggingFace) |
| Frontend | HTML, CSS, JavaScript |
| Language | Python 3.13 |

## Quick Start
`ash
git clone https://github.com/Aryan777827/Streamline_Ai.git
cd Streamline_Ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
`

Open http://localhost:8000 in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Frontend dashboard |
| POST | /upload | Upload video, returns job_id |
| GET | /status/{job_id} | Poll processing progress (0-100%) |
| GET | /results/{job_id} | Get full JSON results |
| GET | /health | API health check |
| GET | /jobs | List all processed jobs |

## Project Structure
`
streamlineai/
├── src/
│   ├── preprocessing/
│   │   ├── video_processor.py     # Frame extraction, video info
│   │   ├── scene_detector.py      # Scene cut detection
│   │   └── audio_processor.py     # Whisper transcription
│   ├── models/
│   │   ├── object_detector.py     # YOLOv8 object detection
│   │   └── sentiment_analyzer.py  # Sentiment scoring
│   └── inference/
│       ├── complete_pipeline.py   # End-to-end orchestration
│       ├── video_analyzer.py      # Video analysis utilities
│       └── recommendation_engine.py # Content recommendations
├── main.py                        # FastAPI backend
├── index.html                     # Frontend dashboard
└── requirements.txt
`

## Development Progress

- Week 1: Video frame extraction + YOLOv8 object detection
- Week 2: Scene detection + key frame extraction
- Week 3: Audio transcription with Whisper
- Week 4: Sentiment analysis
- Week 5: Recommendation engine + video analyzer
- Week 6: Complete pipeline integration
- Week 7: FastAPI backend + frontend dashboard

## Author

Aryan - building AI/ML skills for 2028 tech roles at Google, NVIDIA, Meta

https://github.com/Aryan777827
