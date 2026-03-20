# StreamlineAI — AI-Powered Video Analysis Platform

A full-stack AI video intelligence platform built as a 2028 graduation portfolio project.

## What it does

Upload any video and StreamlineAI automatically runs a complete AI pipeline:

- **Scene Detection** — finds every cut and scene change
- **Object Detection** — tags every object using YOLOv8n (COCO-80 classes)
- **Audio Transcription** — transcribes speech using OpenAI Whisper
- **Sentiment Analysis** — analyses tone of the transcript
- **Web Dashboard** — beautiful UI to explore all results

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Computer Vision:** OpenCV, YOLOv8 (Ultralytics)
- **Speech:** OpenAI Whisper
- **Deep Learning:** PyTorch
- **Frontend:** Vanilla HTML/CSS/JS
- **Language:** Python 3.13

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

## Project Structure
`
streamlineai/
├── src/
│   ├── preprocessing/    # VideoProcessor, SceneDetector, AudioProcessor
│   ├── models/           # ObjectDetector (YOLOv8), SentimentAnalyzer
│   └── inference/        # CompletePipeline — runs everything end to end
├── main.py               # FastAPI backend (upload, status, results endpoints)
├── index.html            # Frontend dashboard UI
└── requirements.txt
`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload video, returns job_id |
| GET | /status/{job_id} | Poll processing progress |
| GET | /results/{job_id} | Get full JSON results |
| GET | / | Frontend dashboard |

## Progress

- Week 1: Video frame extraction + YOLOv8 object detection
- Week 2: Scene detection
- Week 3: Audio transcription (Whisper)
- Week 4: Sentiment analysis
- Week 7: FastAPI backend + full frontend dashboard

## Author

Aryan — building AI/ML skills for 2028 tech roles at Google, NVIDIA, Meta
