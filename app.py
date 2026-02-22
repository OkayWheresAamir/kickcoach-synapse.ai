#DEPENDENCIES:

#for file handling & timestamp
import os
import tempfile
from datetime import datetime

#for trig/hypot
import math

#to save CSV
import csv

#used later for summary in other versions (not strictly in this handler but harmless)
import statistics

#to produce structured logs
import logging

#to create a lock
import threading

#FastAPI IMPORTS:
#to define endpoints that accept multipart uploads, to return HTTP errors, used by middleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
#to return JSON
from fastapi.responses import JSONResponse
#from Starlette used to implement upload-size limiting
from starlette.middleware.base import BaseHTTPMiddleware

#OpenCV: reads video frames from file via VideoCapture
import cv2

#array ops and numeric handling
import numpy as np

#used only for TFLite Interpreter (tf.lite.Interpreter)
import tensorflow as tf


#--------------------
# CONFIG (tweakable) 
#--------------------
MODEL_PATH = "3.tflite"
CONF_THRESH = 0.35 #pose keypoint confidence threshold (0..1) to decide whether to use a keypoint.
KICKING_LEG = "right" #which leg you analyse ("right" or "left").

# DETECTION THRESHOLDS:
KICK_THRESHOLD_PPS = 360 #ankle linear speed threshold in 'pixels/second' used to consider a kick as "fast". (Units: px/sec)
KNEE_ANG_VEL_THRESHOLD = 250 #knee angular velocity threshold in degrees / second to flag an explosive knee movement.
KICK_COOLDOWN_SECS = 0.5 #cooldown duration after a detected kick -> prevents duplicates.
SMOOTH_ALPHA = 0.35 #EMA smoothing factor (0..1). Higher → less smoothing (more weight to new values).
FPS_ESTIMATE = 30.0 #fallback fps if video metadata missing.

# production safety
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  #middleware uses this to reject too-large uploads (200MB).
ALLOWED_MIME_PREFIX = "video/"        #simple MIME type check (video/*).

# panel width (unused in API but we kept for parity)
PANEL_W = 320


#---------
# LOGGING
#---------
#Sets up a basic logger; use logger.info(), logger.error() etc. throughout code for observability.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - KickCoach - %(name)s - %(levelname)s - %(message)s"
)
# logger = logging.getLogger("KickCoach_api")
logger = logging.getLogger(__name__)


#-----------------------------------
# Load TFLite Interpreter + warm-up
#-----------------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#Instantiates the TFLite interpreter and allocates tensors. Doing this at import time (module load) means each process (worker) will load the model once (good).
#IMPORTANT NOTE: in multi-worker setups (gunicorn / *uvicorn* workers), each worker is a separate process and will create its own interpreter copy --- expected.
logger.info("TFLite interpreter loaded and tensors allocated")

# Interpreter concurrency lock (safe-guard if you run single process with threads)
interp_lock = threading.Lock()

# Warm-up: run a dummy inference to reduce first-request latency
try:
    dummy = np.zeros((1, 192, 192, 3), dtype=np.float32)
    with interp_lock:
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        #interp_lock protects the interpreter if you ever run inference from multiple threads inside the same process. TFLite’s Python wrapper is not strictly thread-safe, so the lock prevents concurrent invoke() calls.
        #Warm-up runs a dummy pass to reduce the first-request latency (initial compilation / delegate overhead).
    logger.info("Interpreter warm-up complete")
except Exception as e:
    logger.exception("Interpreter warm-up failed: %s", e)

# ==============================
# Helpers (same as notebook)
# ==============================
EDGES = {  # kept for completeness
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

def scaled_keypoints_from_output(keypoints, frame_shape):
    h, w, _ = frame_shape
    shaped = np.squeeze(keypoints)  # (17,3)
    pixel_kps = np.zeros_like(shaped, dtype=np.float32)
    for i in range(shaped.shape[0]):
        y_norm, x_norm, sc = shaped[i]
        pixel_kps[i, 0] = float(y_norm * h)
        pixel_kps[i, 1] = float(x_norm * w)
        pixel_kps[i, 2] = float(sc)
    return pixel_kps

def angle_between_points(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    BA = a - b
    BC = c - b
    nBA = np.linalg.norm(BA)
    nBC = np.linalg.norm(BC)
    if nBA < 1e-6 or nBC < 1e-6:
        return np.nan
    cos_ang = np.dot(BA, BC) / (nBA * nBC)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))

def trunk_tilt_signed_degrees(shoulder_l, shoulder_r, hip_l, hip_r):
    try:
        sh_mid_x = (shoulder_l[0] + shoulder_r[0]) / 2.0
        sh_mid_y = (shoulder_l[1] + shoulder_r[1]) / 2.0
        hip_mid_x = (hip_l[0] + hip_r[0]) / 2.0
        hip_mid_y = (hip_l[1] + hip_r[1]) / 2.0
    except Exception:
        return np.nan
    dx = sh_mid_x - hip_mid_x
    dy = sh_mid_y - hip_mid_y
    v_norm = math.hypot(dx, dy)
    if v_norm < 1e-6:
        return np.nan
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    return float(angle_deg)

# Simple EMA smoother
class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return self.v
        if self.v is None:
            self.v = float(x)
        else:
            self.v = float(self.alpha * x + (1 - self.alpha) * self.v)
        return self.v

def nan_to_none(x):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except Exception:
        pass
    return x

def save_final_snapshot_csv(snapshot, fname="final_kick_features.csv"):
    out = {k: ("" if v is None else v) for k,v in snapshot.items()}
    out["captured_at"] = datetime.now().isoformat()
    fieldnames = list(out.keys())
    write_header = not os.path.exists(fname)
    with open(fname, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(out)

# ==============================
# Middleware: limit upload size using Content-Length header (best-effort)
# ==============================
class MaxSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body: int):
        super().__init__(app)
        self.max_body = max_body

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > self.max_body:
                    logger.warning("Rejected request: content-length %s exceeds %s", cl, self.max_body)
                    raise HTTPException(status_code=413, detail="Upload too large")
            except ValueError:
                # ignore invalid content-length header
                pass
        return await call_next(request)

# ==============================
# FastAPI app
# ==============================
app = FastAPI(title="MoveNet Kick Analyzer")
app.add_middleware(MaxSizeMiddleware, max_body=MAX_UPLOAD_BYTES)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    # simple readiness: interpreter should be loaded
    return {"status": "ready", "model_loaded": True}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a video file (multipart form). The endpoint processes the video
    and returns the final kick snapshot captured at the first detected kick.
    """
    # basic mime check
    ctype = (file.content_type or "").lower()
    if not ctype.startswith(ALLOWED_MIME_PREFIX):
        logger.warning("Rejecting file with content-type: %s", ctype)
        raise HTTPException(status_code=415, detail="Unsupported file type; expect video/*")

    # save uploaded file temporarily (streamed write)
    suffix = os.path.splitext(file.filename or "upload")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        # stream write to avoid building huge memory objects (UploadFile.file is a SpooledTemporaryFile)
        # but here we use read() (fine under middleware limit). If you need robust streaming, read chunks:
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload too large")
        tmp.write(content)
        tmp.flush()
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            logger.error("Cannot open video file %s", tmp.name)
            raise HTTPException(status_code=400, detail="Could not open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_ESTIMATE
        kick_cooldown_frames = max(1, int(round(KICK_COOLDOWN_SECS * fps)))
        kick_cooldown = 0

        knee_ema = EMA(SMOOTH_ALPHA)
        trunk_ema = EMA(SMOOTH_ALPHA)

        prev_ankle = None
        prev_knee_angle = None
        final_snapshot = None
        frame_idx = 0

        # joint indices
        LHIP, RHIP = 11, 12
        LKNEE, RKNEE = 13, 14
        LANKLE, RANKLE = 15, 16
        LSH, RSH = 5, 6

        # process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape

            try:
                # Prepare input for MoveNet (normalized frame)
                img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
                input_image = tf.cast(img, dtype=tf.float32)

                # Run inference inside lock for thread-safety
                with interp_lock:
                    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
                    interpreter.invoke()
                    kps_norm = interpreter.get_tensor(output_details[0]['index'])
            except Exception as inf_e:
                logger.exception("TFLite inference error at frame %d: %s", frame_idx, inf_e)
                # continue processing frames (or optionally break)
                frame_idx += 1
                continue

            pixel_kps = scaled_keypoints_from_output(kps_norm, frame.shape)

            # kicking leg
            if KICKING_LEG.lower().startswith("r"):
                hip_idx, knee_idx, ankle_idx = RHIP, RKNEE, RANKLE
            else:
                hip_idx, knee_idx, ankle_idx = LHIP, LKNEE, LANKLE

            def to_xy_conf(idx):
                y, x, c = pixel_kps[idx]
                return (float(x), float(y)), float(c)

            hip_pt, hip_conf = to_xy_conf(hip_idx)
            knee_pt, knee_conf = to_xy_conf(knee_idx)
            ankle_pt, ankle_conf = to_xy_conf(ankle_idx)

            # knee angle
            knee_angle = np.nan
            if min(hip_conf, knee_conf, ankle_conf) >= CONF_THRESH:
                knee_angle = angle_between_points(hip_pt, knee_pt, ankle_pt)

            # trunk value
            sh_l_pt, sh_l_conf = to_xy_conf(LSH)
            sh_r_pt, sh_r_conf = to_xy_conf(RSH)
            trunk_val = np.nan
            try:
                hip_l_pt, hip_l_conf = to_xy_conf(LHIP)
                hip_r_pt, hip_r_conf = to_xy_conf(RHIP)
                if min(sh_l_conf, sh_r_conf, hip_l_conf, hip_r_conf) >= CONF_THRESH:
                    trunk_val = trunk_tilt_signed_degrees(sh_l_pt, sh_r_pt, hip_l_pt, hip_r_pt)
            except Exception:
                trunk_val = np.nan

            # ankle speed
            ankle_speed_pps = np.nan
            if prev_ankle is not None and ankle_conf >= CONF_THRESH:
                dx = ankle_pt[0] - prev_ankle[0]
                dy = ankle_pt[1] - prev_ankle[1]
                ankle_speed_pxpf = math.hypot(dx, dy)
                ankle_speed_pps = ankle_speed_pxpf * fps
            prev_ankle = ankle_pt

            # knee angular vel
            knee_ang_vel_dps = np.nan
            if prev_knee_angle is not None and not np.isnan(knee_angle):
                knee_ang_vel_dps = (knee_angle - prev_knee_angle) * fps
            prev_knee_angle = knee_angle if not np.isnan(knee_angle) else prev_knee_angle

            # smoothing
            sm_knee = knee_ema.update(knee_angle)
            sm_trunk = trunk_ema.update(trunk_val)

            # detection logic
            if kick_cooldown > 0:
                kick_cooldown -= 1

            is_fast = isinstance(ankle_speed_pps, (int, float)) and not np.isnan(ankle_speed_pps) and ankle_speed_pps > KICK_THRESHOLD_PPS
            is_explosive = isinstance(knee_ang_vel_dps, (int, float)) and not np.isnan(knee_ang_vel_dps) and abs(knee_ang_vel_dps) > KNEE_ANG_VEL_THRESHOLD

            if (is_fast and is_explosive and kick_cooldown == 0 and final_snapshot is None):
                final_snapshot = {
                    "frame_number": int(frame_idx),
                    "knee_angle": nan_to_none(sm_knee),
                    "trunk": nan_to_none(sm_trunk),
                    "ankle_speed_pps": nan_to_none(ankle_speed_pps),
                    "knee_ang_vel_dps": nan_to_none(knee_ang_vel_dps)
                }
                # best-effort persist
                try:
                    save_final_snapshot_csv(final_snapshot, fname="final_kick_features.csv")
                except Exception:
                    logger.exception("Failed to save final snapshot CSV")

                kick_cooldown = kick_cooldown_frames

            frame_idx += 1

        cap.release()

        if final_snapshot is None:
            logger.info("No kick detected in uploaded video")
            return JSONResponse({"status": "no_kick_detected", "final_kick": None}, status_code=200)
        else:
            logger.info("Kick detected at frame %s", final_snapshot.get("frame_number"))
            return JSONResponse({"status": "ok", "final_kick": final_snapshot}, status_code=200)

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

# optional root
@app.get("/")
def root():
    return {"status": "ready", "note": "POST video file to /analyze (form field 'file')"}