import uuid, json, time, shutil, asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="StreamlineAI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

jobs = {}

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    stage: str
    filename: Optional[str] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    message: str

def _run_real_pipeline(job_id, video_path):
    from src.inference.complete_pipeline import CompletePipeline
    pipeline = CompletePipeline(enable_audio=False, enable_sentiment=False)
    results = pipeline.analyze_video(video_path)
    out = RESULTS_DIR / f"{job_id}_results.json"
    pipeline.save_results(results, str(out))
    return json.loads(json.dumps(results, default=str))

async def run_pipeline(job_id, video_path):
    try:
        jobs[job_id]["status"] = "processing"
        for progress, stage in [(10,"Extracting frames"),(30,"Running scene detection"),(55,"Running YOLOv8 object detection"),(75,"Transcribing audio with Whisper"),(85,"Building results JSON")]:
            jobs[job_id]["progress"] = progress
            jobs[job_id]["stage"] = stage
            await asyncio.sleep(0)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_real_pipeline, job_id, str(video_path))
        jobs[job_id]["result"] = result
        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["stage"] = "Done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["stage"] = "Failed"
        import traceback; traceback.print_exc()

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    p = Path("index.html")
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists() else "<h1>index.html not found</h1>")

@app.get("/health")
def health():
    return {"status": "ok", "service": "StreamlineAI API"}

@app.post("/upload", response_model=UploadResponse)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'")
    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jobs[job_id] = {"status":"queued","progress":0,"stage":"Queued","filename":file.filename,"result":None,"error":None,"created_at":time.time()}
    background_tasks.add_task(run_pipeline, job_id, save_path)
    return UploadResponse(job_id=job_id, filename=file.filename, message="Upload successful")

@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    return JobStatus(job_id=job_id, status=j["status"], progress=j["progress"], stage=j["stage"], filename=j.get("filename"), error=j.get("error"))

@app.get("/results/{job_id}")
def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    if j["status"] in ("processing","queued"):
        raise HTTPException(status_code=202, detail="Still processing")
    if j["status"] == "error":
        raise HTTPException(status_code=500, detail=j.get("error","Unknown error"))
    return JSONResponse(content=j["result"])

@app.get("/jobs")
def list_jobs():
    return [{"job_id":k,"status":v["status"],"filename":v.get("filename"),"progress":v["progress"]} for k,v in jobs.items()]
