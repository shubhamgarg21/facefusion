"""
FaceFusion API Server

This FastAPI server provides RESTful endpoints for all FaceFusion processors:

MAIN PROCESSING ENDPOINTS:
- POST /face-swap: Face swapping with source and target files
- POST /face-enhance: Face enhancement with source and target files  
- POST /frame-enhance: Frame/video enhancement (no source needed)
- POST /frame-colorize: Frame/video colorization (no source needed)
- POST /face-debug: Face debugging with source and target files

BATCH PROCESSING:
- POST /batch-face-swap: Process multiple target files with one source

JOB MANAGEMENT:
- GET /jobs/{job_id}/status: Get job status
- GET /jobs: List all jobs
- GET /jobs/{job_id}/download: Download completed results
- DELETE /jobs/{job_id}: Delete job and files

MODEL/OPTION ENDPOINTS:
- GET /models/face-detector: Available face detector models
- GET /models/face-swapper: Available face swapper models
- GET /models/face-enhancer: Available face enhancer models
- GET /models/frame-enhancer: Available frame enhancer models
- GET /models/frame-colorizer: Available frame colorizer models
- GET /models/frame-colorizer-sizes: Available colorizer sizes
- GET /models/face-debugger-items: Available debugger items
- GET /processors: List all available processors

HEALTH/INFO:
- GET /: API info
- GET /health: Health check

All processing endpoints return a job_id for tracking progress.
Use the job management endpoints to monitor and retrieve results.
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import shutil
import asyncio
from datetime import datetime

app = FastAPI(
    title="FaceFusion API Server",
    description="API server for FaceFusion face swapping using CLI interface",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage
jobs: Dict[str, Dict[str, Any]] = {}

# Models for API requests
class FaceSwapRequest(BaseModel):
    processors: List[str] = ["face_swapper"]
    face_detector_model: Optional[str] = "yolo_face"
    face_detector_score: Optional[float] = 0.5
    face_swapper_model: Optional[str] = None
    face_swapper_pixel_boost: Optional[str] = None
    output_video_quality: Optional[int] = 80
    output_image_quality: Optional[int] = 80
    trim_frame_start: Optional[int] = None
    trim_frame_end: Optional[int] = None
    keep_temp: Optional[bool] = False

class FaceEnhanceRequest(BaseModel):
    processors: List[str] = ["face_enhancer"]
    face_detector_model: Optional[str] = "yolo_face"
    face_detector_score: Optional[float] = 0.5
    face_enhancer_model: Optional[str] = "gfpgan_1.4"
    face_enhancer_blend: Optional[int] = 80
    face_enhancer_weight: Optional[float] = 1.0
    output_video_quality: Optional[int] = 80
    output_image_quality: Optional[int] = 80
    trim_frame_start: Optional[int] = None
    trim_frame_end: Optional[int] = None
    keep_temp: Optional[bool] = False

class FrameEnhanceRequest(BaseModel):
    processors: List[str] = ["frame_enhancer"]
    frame_enhancer_model: Optional[str] = "span_kendata_x4"
    frame_enhancer_blend: Optional[int] = 80
    output_video_quality: Optional[int] = 80
    output_image_quality: Optional[int] = 80
    trim_frame_start: Optional[int] = None
    trim_frame_end: Optional[int] = None
    keep_temp: Optional[bool] = False

class FrameColorizerRequest(BaseModel):
    processors: List[str] = ["frame_colorizer"]
    frame_colorizer_model: Optional[str] = "ddcolor"
    frame_colorizer_size: Optional[str] = "256x256"
    frame_colorizer_blend: Optional[int] = 100
    output_video_quality: Optional[int] = 80
    output_image_quality: Optional[int] = 80
    trim_frame_start: Optional[int] = None
    trim_frame_end: Optional[int] = None
    keep_temp: Optional[bool] = False

class FaceDebugRequest(BaseModel):
    processors: List[str] = ["face_debugger"]
    face_detector_model: Optional[str] = "yolo_face"
    face_detector_score: Optional[float] = 0.5
    face_debugger_items: List[str] = ["face-landmark-5/68", "face-mask"]
    output_video_quality: Optional[int] = 80
    output_image_quality: Optional[int] = 80
    trim_frame_start: Optional[int] = None
    trim_frame_end: Optional[int] = None
    keep_temp: Optional[bool] = False

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    output_file: Optional[str] = None

# Helper functions
def create_job_id() -> str:
    return str(uuid.uuid4())

def save_uploaded_file(upload_file: UploadFile, job_dir: str, prefix: str) -> str:
    """Save uploaded file and return the path"""
    file_extension = os.path.splitext(upload_file.filename)[1]
    file_path = os.path.join(job_dir, f"{prefix}{file_extension}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path

def build_cli_command(
    source_path: str,
    target_path: str,
    output_path: str,
    options: FaceSwapRequest,
    jobs_path: str
) -> List[str]:
    """Build the CLI command for face swapping"""
    
    cmd = [
        "python", "facefusion.py", "headless-run",
        "--jobs-path", jobs_path,
        "--processors", " ".join(options.processors),
        "-s", source_path,
        "-t", target_path,
        "-o", output_path
    ]
    
    # Add optional parameters
    if options.face_detector_model:
        cmd.extend(["--face-detector-model", options.face_detector_model])
    
    if options.face_detector_score is not None:
        cmd.extend(["--face-detector-score", str(options.face_detector_score)])
    
    if options.face_swapper_model:
        cmd.extend(["--face-swapper-model", options.face_swapper_model])
    
    if options.face_swapper_pixel_boost:
        cmd.extend(["--face-swapper-pixel-boost", options.face_swapper_pixel_boost])
    
    if options.output_video_quality is not None:
        cmd.extend(["--output-video-quality", str(options.output_video_quality)])
    
    if options.output_image_quality is not None:
        cmd.extend(["--output-image-quality", str(options.output_image_quality)])
    
    if options.trim_frame_start is not None:
        cmd.extend(["--trim-frame-start", str(options.trim_frame_start)])
    
    if options.trim_frame_end is not None:
        cmd.extend(["--trim-frame-end", str(options.trim_frame_end)])
    
    if options.keep_temp:
        cmd.append("--keep-temp")
    
    return cmd

def build_general_cli_command(
    source_path: Optional[str],
    target_path: str,
    output_path: str,
    options,
    jobs_path: str
) -> List[str]:
    """Build the CLI command for any processor type"""
    
    cmd = [
        "python", "facefusion.py", "headless-run",
        "--jobs-path", jobs_path,
        "--processors", " ".join(options.processors)
    ]
    
    # Add source if provided (not needed for frame processors)
    if source_path:
        cmd.extend(["-s", source_path])
    
    cmd.extend(["-t", target_path, "-o", output_path])
    
    # Add common optional parameters
    if hasattr(options, 'face_detector_model') and options.face_detector_model:
        cmd.extend(["--face-detector-model", options.face_detector_model])
    
    if hasattr(options, 'face_detector_score') and options.face_detector_score is not None:
        cmd.extend(["--face-detector-score", str(options.face_detector_score)])
    
    # Face swapper parameters
    if hasattr(options, 'face_swapper_model') and options.face_swapper_model:
        cmd.extend(["--face-swapper-model", options.face_swapper_model])
    
    if hasattr(options, 'face_swapper_pixel_boost') and options.face_swapper_pixel_boost:
        cmd.extend(["--face-swapper-pixel-boost", options.face_swapper_pixel_boost])
    
    # Face enhancer parameters
    if hasattr(options, 'face_enhancer_model') and options.face_enhancer_model:
        cmd.extend(["--face-enhancer-model", options.face_enhancer_model])
    
    if hasattr(options, 'face_enhancer_blend') and options.face_enhancer_blend is not None:
        cmd.extend(["--face-enhancer-blend", str(options.face_enhancer_blend)])
    
    if hasattr(options, 'face_enhancer_weight') and options.face_enhancer_weight is not None:
        cmd.extend(["--face-enhancer-weight", str(options.face_enhancer_weight)])
    
    # Frame enhancer parameters
    if hasattr(options, 'frame_enhancer_model') and options.frame_enhancer_model:
        cmd.extend(["--frame-enhancer-model", options.frame_enhancer_model])
    
    if hasattr(options, 'frame_enhancer_blend') and options.frame_enhancer_blend is not None:
        cmd.extend(["--frame-enhancer-blend", str(options.frame_enhancer_blend)])
    
    # Frame colorizer parameters
    if hasattr(options, 'frame_colorizer_model') and options.frame_colorizer_model:
        cmd.extend(["--frame-colorizer-model", options.frame_colorizer_model])
    
    if hasattr(options, 'frame_colorizer_size') and options.frame_colorizer_size:
        cmd.extend(["--frame-colorizer-size", options.frame_colorizer_size])
    
    if hasattr(options, 'frame_colorizer_blend') and options.frame_colorizer_blend is not None:
        cmd.extend(["--frame-colorizer-blend", str(options.frame_colorizer_blend)])
    
    # Face debugger parameters
    if hasattr(options, 'face_debugger_items') and options.face_debugger_items:
        cmd.extend(["--face-debugger-items"] + options.face_debugger_items)
    
    # Output quality parameters
    if hasattr(options, 'output_video_quality') and options.output_video_quality is not None:
        cmd.extend(["--output-video-quality", str(options.output_video_quality)])
    
    if hasattr(options, 'output_image_quality') and options.output_image_quality is not None:
        cmd.extend(["--output-image-quality", str(options.output_image_quality)])
    
    # Trim parameters
    if hasattr(options, 'trim_frame_start') and options.trim_frame_start is not None:
        cmd.extend(["--trim-frame-start", str(options.trim_frame_start)])
    
    if hasattr(options, 'trim_frame_end') and options.trim_frame_end is not None:
        cmd.extend(["--trim-frame-end", str(options.trim_frame_end)])
    
    if hasattr(options, 'keep_temp') and options.keep_temp:
        cmd.append("--keep-temp")
    
    return cmd

async def run_face_swap_job(job_id: str, cmd: List[str], output_path: str):
    """Run face swap job asynchronously"""
    try:
        jobs[job_id]["status"] = "processing"
        
        # Run the CLI command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            jobs[job_id]["output_file"] = output_path
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error_message"] = stderr.decode()
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

# API Endpoints

@app.get("/")
async def root():
    return {"message": "FaceFusion API Server", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/face-swap")
async def face_swap(
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(..., description="Source face image"),
    target_file: UploadFile = File(..., description="Target image or video"),
    processors: str = Form(default="face_swapper"),
    face_detector_model: str = Form(default="yolo_face"),
    face_detector_score: float = Form(default=0.5),
    face_swapper_model: Optional[str] = Form(default=None),
    face_swapper_pixel_boost: Optional[str] = Form(default=None),
    output_video_quality: int = Form(default=80),
    output_image_quality: int = Form(default=80),
    trim_frame_start: Optional[int] = Form(default=None),
    trim_frame_end: Optional[int] = Form(default=None),
    keep_temp: bool = Form(default=False)
):
    """
    Perform face swapping with uploaded files
    """
    job_id = create_job_id()
    
    # Create job directory
    job_dir = os.path.join("jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        # Save uploaded files
        source_path = save_uploaded_file(source_image, job_dir, "source")
        target_path = save_uploaded_file(target_file, job_dir, "target")
        
        # Determine output file extension based on target
        target_ext = os.path.splitext(target_file.filename)[1]
        output_path = os.path.join(job_dir, f"output{target_ext}")
        
        # Create request options
        options = FaceSwapRequest(
            processors=processors.split(","),
            face_detector_model=face_detector_model,
            face_detector_score=face_detector_score,
            face_swapper_model=face_swapper_model,
            face_swapper_pixel_boost=face_swapper_pixel_boost,
            output_video_quality=output_video_quality,
            output_image_quality=output_image_quality,
            trim_frame_start=trim_frame_start,
            trim_frame_end=trim_frame_end,
            keep_temp=keep_temp
        )
        
        # Build CLI command
        cmd = build_cli_command(source_path, target_path, output_path, options, ".jobs")
        
        # Store job info
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "source_file": source_image.filename,
            "target_file": target_file.filename,
            "command": " ".join(cmd)
        }
        
        # Run job in background
        background_tasks.add_task(run_face_swap_job, job_id, cmd, output_path)
        
        return {"job_id": job_id, "status": "pending", "message": "Face swap job started"}
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the status of a specific job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {"jobs": list(jobs.values())}

@app.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download the result file of a completed job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    output_file = job.get("output_file")
    if not output_file or not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_file,
        media_type="application/octet-stream",
        filename=os.path.basename(output_file)
    )

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove job directory
    job_dir = os.path.join("jobs", job_id)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    
    # Remove from jobs dict
    del jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/models/face-detector")
async def get_face_detector_models():
    """Get available face detector models"""
    return {
        "models": [
            "yolo_face",
            "retinaface",
            "scrfd",
            "yunet"
        ]
    }

@app.get("/models/face-swapper")
async def get_face_swapper_models():
    """Get available face swapper models"""
    return {
        "models": [
            "blendswap_256",
            "ghost_1_256", 
            "ghost_2_256",
            "ghost_3_256",
            "inswapper_128",
            "inswapper_128_fp16",
            "simswap_256",
            "simswap_512_unofficial",
            "uniface_256"
        ]
    }

@app.get("/models/face-enhancer")
async def get_face_enhancer_models():
    """Get available face enhancer models"""
    return {
        "models": [
            "codeformer",
            "gfpgan_1.2", 
            "gfpgan_1.3",
            "gfpgan_1.4",
            "gpen_bfr_256", 
            "gpen_bfr_512",
            "gpen_bfr_1024",
            "gpen_bfr_2048",
            "restoreformer_plus_plus"
        ]
    }

@app.get("/models/frame-enhancer")
async def get_frame_enhancer_models():
    """Get available frame enhancer models"""
    return {
        "models": [
            "clear_reality_x4",
            "lsdir_x4",
            "nomos8k_sc_x4",
            "real_esrgan_x2",
            "real_esrgan_x2_fp16",
            "real_esrgan_x4",
            "real_esrgan_x4_fp16",
            "real_esrgan_x8",
            "real_esrgan_x8_fp16",
            "real_hatgan_x4",
            "real_web_photo_x4",
            "realistic_rescaler_x4",
            "remacri_x4",
            "siax_x4",
            "span_kendata_x4",
            "swin2_sr_x4",
            "ultra_sharp_x4"
        ]
    }

@app.get("/models/frame-colorizer")
async def get_frame_colorizer_models():
    """Get available frame colorizer models"""
    return {
        "models": [
            "ddcolor",
            "ddcolor_artistic",
            "deoldify",
            "deoldify_artistic",
            "deoldify_stable"
        ]
    }

@app.get("/models/frame-colorizer-sizes")
async def get_frame_colorizer_sizes():
    """Get available frame colorizer sizes"""
    return {
        "sizes": [
            "192x192",
            "256x256",
            "384x384",
            "512x512"
        ]
    }

@app.get("/models/face-debugger-items")
async def get_face_debugger_items():
    """Get available face debugger items"""
    return {
        "items": [
            "bounding-box",
            "face-landmark-5",
            "face-landmark-5/68",
            "face-landmark-68",
            "face-landmark-68/5",
            "face-mask",
            "face-detector-score",
            "face-landmarker-score",
            "age",
            "gender",
            "race"
        ]
    }

@app.get("/processors")
async def get_available_processors():
    """Get available processors"""
    return {
        "processors": [
            "face_swapper",
            "face_enhancer", 
            "frame_enhancer",
            "face_debugger",
            "frame_colorizer"
        ]
    }

# Batch processing endpoint
@app.post("/batch-face-swap")
async def batch_face_swap(
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(...),
    target_files: List[UploadFile] = File(...),
    processors: str = Form(default="face_swapper"),
    face_detector_model: str = Form(default="yolo_face"),
    face_detector_score: float = Form(default=0.5)
):
    """
    Perform face swapping on multiple target files
    """
    batch_id = create_job_id()
    job_ids = []
    
    for i, target_file in enumerate(target_files):
        job_id = create_job_id()
        job_dir = os.path.join("jobs", job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        try:
            # Save files
            source_path = save_uploaded_file(source_image, job_dir, "source")
            target_path = save_uploaded_file(target_file, job_dir, f"target_{i}")
            
            target_ext = os.path.splitext(target_file.filename)[1]
            output_path = os.path.join(job_dir, f"output{target_ext}")
            
            # Create options
            options = FaceSwapRequest(
                processors=processors.split(","),
                face_detector_model=face_detector_model,
                face_detector_score=face_detector_score
            )
            
            # Build command
            cmd = build_cli_command(source_path, target_path, output_path, options, ".jobs")
            
            # Store job
            jobs[job_id] = {
                "job_id": job_id,
                "batch_id": batch_id,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "source_file": source_image.filename,
                "target_file": target_file.filename
            }
            
            # Start job
            background_tasks.add_task(run_face_swap_job, job_id, cmd, output_path)
            job_ids.append(job_id)
            
        except Exception as e:
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir)
            continue
    
    return {
        "batch_id": batch_id,
        "job_ids": job_ids,
        "message": f"Started {len(job_ids)} face swap jobs"
    }

@app.post("/face-enhance")
async def face_enhance(
    background_tasks: BackgroundTasks,
    target_file: UploadFile = File(..., description="Target image or video to enhance"),
    source_image: Optional[UploadFile] = File(None, description="Optional source face image for reference"),
    processors: str = Form(default="face_enhancer"),
    face_detector_model: str = Form(default="yolo_face"),
    face_detector_score: float = Form(default=0.5),
    face_enhancer_model: str = Form(default="gfpgan_1.4"),
    face_enhancer_blend: int = Form(default=80),
    face_enhancer_weight: float = Form(default=1.0),
    output_video_quality: int = Form(default=80),
    output_image_quality: int = Form(default=80),
    trim_frame_start: Optional[int] = Form(default=None),
    trim_frame_end: Optional[int] = Form(default=None),
    keep_temp: bool = Form(default=False)
):
    """
    Perform face enhancement with uploaded files
    Source image is optional - if provided, only enhances faces similar to reference
    """
    job_id = create_job_id()
    
    # Create job directory
    job_dir = os.path.join("jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        # Save target file (required)
        target_path = save_uploaded_file(target_file, job_dir, "target")
        
        # Save source file if provided (optional)
        source_path = None
        if source_image:
            source_path = save_uploaded_file(source_image, job_dir, "source")
        
        # Determine output file extension based on target
        target_ext = os.path.splitext(target_file.filename)[1]
        output_path = os.path.join(job_dir, f"enhanced{target_ext}")
        
        # Create request options
        options = FaceEnhanceRequest(
            processors=processors.split(","),
            face_detector_model=face_detector_model,
            face_detector_score=face_detector_score,
            face_enhancer_model=face_enhancer_model,
            face_enhancer_blend=face_enhancer_blend,
            face_enhancer_weight=face_enhancer_weight,
            output_video_quality=output_video_quality,
            output_image_quality=output_image_quality,
            trim_frame_start=trim_frame_start,
            trim_frame_end=trim_frame_end,
            keep_temp=keep_temp
        )
        
        # Build CLI command (source_path can be None)
        cmd = build_general_cli_command(source_path, target_path, output_path, options, ".jobs")
        
        # Store job info
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "source_file": source_image.filename if source_image else None,
            "target_file": target_file.filename,
            "command": " ".join(cmd)
        }
        
        # Run job in background
        background_tasks.add_task(run_face_swap_job, job_id, cmd, output_path)
        
        return {"job_id": job_id, "status": "pending", "message": "Face enhancement job started"}
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/frame-enhance")
async def frame_enhance(
    background_tasks: BackgroundTasks,
    target_file: UploadFile = File(..., description="Target image or video to enhance"),
    processors: str = Form(default="frame_enhancer"),
    frame_enhancer_model: str = Form(default="span_kendata_x4"),
    frame_enhancer_blend: int = Form(default=80),
    output_video_quality: int = Form(default=80),
    output_image_quality: int = Form(default=80),
    trim_frame_start: Optional[int] = Form(default=None),
    trim_frame_end: Optional[int] = Form(default=None),
    keep_temp: bool = Form(default=False)
):
    """
    Perform frame enhancement with uploaded file
    """
    job_id = create_job_id()
    
    # Create job directory
    job_dir = os.path.join("jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        # Save uploaded file
        target_path = save_uploaded_file(target_file, job_dir, "target")
        
        # Determine output file extension based on target
        target_ext = os.path.splitext(target_file.filename)[1]
        output_path = os.path.join(job_dir, f"enhanced{target_ext}")
        
        # Create request options
        options = FrameEnhanceRequest(
            processors=processors.split(","),
            frame_enhancer_model=frame_enhancer_model,
            frame_enhancer_blend=frame_enhancer_blend,
            output_video_quality=output_video_quality,
            output_image_quality=output_image_quality,
            trim_frame_start=trim_frame_start,
            trim_frame_end=trim_frame_end,
            keep_temp=keep_temp
        )
        
        # Build CLI command (no source needed for frame enhancement)
        cmd = build_general_cli_command(None, target_path, output_path, options, ".jobs")
        
        # Store job info
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "target_file": target_file.filename,
            "command": " ".join(cmd)
        }
        
        # Run job in background
        background_tasks.add_task(run_face_swap_job, job_id, cmd, output_path)
        
        return {"job_id": job_id, "status": "pending", "message": "Frame enhancement job started"}
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/frame-colorize")
async def frame_colorize(
    background_tasks: BackgroundTasks,
    target_file: UploadFile = File(..., description="Target image or video to colorize"),
    processors: str = Form(default="frame_colorizer"),
    frame_colorizer_model: str = Form(default="ddcolor"),
    frame_colorizer_size: str = Form(default="256x256"),
    frame_colorizer_blend: int = Form(default=100),
    output_video_quality: int = Form(default=80),
    output_image_quality: int = Form(default=80),
    trim_frame_start: Optional[int] = Form(default=None),
    trim_frame_end: Optional[int] = Form(default=None),
    keep_temp: bool = Form(default=False)
):
    """
    Perform frame colorization with uploaded file
    """
    job_id = create_job_id()
    
    # Create job directory
    job_dir = os.path.join("jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        # Save uploaded file
        target_path = save_uploaded_file(target_file, job_dir, "target")
        
        # Determine output file extension based on target
        target_ext = os.path.splitext(target_file.filename)[1]
        output_path = os.path.join(job_dir, f"colorized{target_ext}")
        
        # Create request options
        options = FrameColorizerRequest(
            processors=processors.split(","),
            frame_colorizer_model=frame_colorizer_model,
            frame_colorizer_size=frame_colorizer_size,
            frame_colorizer_blend=frame_colorizer_blend,
            output_video_quality=output_video_quality,
            output_image_quality=output_image_quality,
            trim_frame_start=trim_frame_start,
            trim_frame_end=trim_frame_end,
            keep_temp=keep_temp
        )
        
        # Build CLI command (no source needed for frame colorization)
        cmd = build_general_cli_command(None, target_path, output_path, options, ".jobs")
        
        # Store job info
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "target_file": target_file.filename,
            "command": " ".join(cmd)
        }
        
        # Run job in background
        background_tasks.add_task(run_face_swap_job, job_id, cmd, output_path)
        
        return {"job_id": job_id, "status": "pending", "message": "Frame colorization job started"}
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/face-debug")
async def face_debug(
    background_tasks: BackgroundTasks,
    target_file: UploadFile = File(..., description="Target image or video to debug"),
    source_image: Optional[UploadFile] = File(None, description="Optional source face image for reference"),
    processors: str = Form(default="face_debugger"),
    face_detector_model: str = Form(default="yolo_face"),
    face_detector_score: float = Form(default=0.5),
    face_debugger_items: str = Form(default="face-landmark-5/68,face-mask"),
    output_video_quality: int = Form(default=80),
    output_image_quality: int = Form(default=80),
    trim_frame_start: Optional[int] = Form(default=None),
    trim_frame_end: Optional[int] = Form(default=None),
    keep_temp: bool = Form(default=False)
):
    """
    Perform face debugging with uploaded files
    Source image is optional - if provided, only debugs faces similar to reference
    """
    job_id = create_job_id()
    
    # Create job directory
    job_dir = os.path.join("jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        # Save target file (required)
        target_path = save_uploaded_file(target_file, job_dir, "target")
        
        # Save source file if provided (optional)
        source_path = None
        if source_image:
            source_path = save_uploaded_file(source_image, job_dir, "source")
        
        # Determine output file extension based on target
        target_ext = os.path.splitext(target_file.filename)[1]
        output_path = os.path.join(job_dir, f"debug{target_ext}")
        
        # Create request options
        options = FaceDebugRequest(
            processors=processors.split(","),
            face_detector_model=face_detector_model,
            face_detector_score=face_detector_score,
            face_debugger_items=face_debugger_items.split(","),
            output_video_quality=output_video_quality,
            output_image_quality=output_image_quality,
            trim_frame_start=trim_frame_start,
            trim_frame_end=trim_frame_end,
            keep_temp=keep_temp
        )
        
        # Build CLI command (source_path can be None)
        cmd = build_general_cli_command(source_path, target_path, output_path, options, ".jobs")
        
        # Store job info
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "source_file": source_image.filename if source_image else None,
            "target_file": target_file.filename,
            "command": " ".join(cmd)
        }
        
        # Run job in background
        background_tasks.add_task(run_face_swap_job, job_id, cmd, output_path)
        
        return {"job_id": job_id, "status": "pending", "message": "Face debugging job started"}
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    os.makedirs("jobs", exist_ok=True)
    os.makedirs(".jobs", exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)