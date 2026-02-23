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

# Warm-up: run a dummy inference to reduce first-request latency (because the FIRST inference is Always Slower)
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

#---------
# HELPERS
#---------
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


#Normalized → Pixel Coordinates
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
#MoveNet output is normalized (y, x, score) in [0..1]; this function converts to pixel coordinates [y_px, x_px, score] given frame height/width.
#NOTE: it keeps order (y, x, score) for consistency with earlier code.

def angle_between_points(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    BA = a - b
    BC = c - b

    nBA = np.linalg.norm(BA) # sqroot(B^2 + A^2)
    nBC = np.linalg.norm(BC)

    if nBA < 1e-6 or nBC < 1e-6:
        return np.nan
    
    cos_ang = np.dot(BA, BC) / (nBA * nBC) #cosine of the angle

    cos_ang = np.clip(cos_ang, -1.0, 1.0) #Limits a value within a range.

    return float(np.degrees(np.arccos(cos_ang))) #arccos() returns angle in radians, not degrees. and then Converts radians → degrees.
#Computes the angle at point b formed by segments b->a and b->c using the cosine rule.
#Returns angle in degrees (0..180). Returns np.nan for degenerate cases (very small segment).

def trunk_tilt_signed_degrees(shoulder_l, shoulder_r, hip_l, hip_r):
    try:
        #Shoulder Midpoint: Represent Upper Torso
        sh_mid_x = (shoulder_l[0] + shoulder_r[0]) / 2.0
        sh_mid_y = (shoulder_l[1] + shoulder_r[1]) / 2.0

        #Hip Midpoint: Represent Lower Torso
        hip_mid_x = (hip_l[0] + hip_r[0]) / 2.0
        hip_mid_y = (hip_l[1] + hip_r[1]) / 2.0

        #Connecting(Shoulder + Hip) their midpoints gives the SPINE DIRECTION
        #So we're building the trunk vector.

    except Exception:
        return np.nan

    #Vector from hips → shoulders:
    dx = sh_mid_x - hip_mid_x
    dy = sh_mid_y - hip_mid_y

    #Computes vector magnitude safely:
    v_norm = math.hypot(dx, dy)

    #Avoids division errors / unstable angle if shoulders and hips collapse to same point
    if v_norm < 1e-6:
        return np.nan
    
    angle_rad = math.atan2(dx, -dy) #since, In images: x-increases → Right & y-increases → DOWNWARD

    '''
    θ=atan2(dx,−dy)
    Means:
    - Reference axis = Vertical
    - Positive angle = Right lean
    - Negative angle = Left lean
    '''
    angle_deg = math.degrees(angle_rad)
    return float(angle_deg)
#Computes a signed trunk tilt angle:
# - v = shoulder_mid - hip_mid.
# - Uses atan2(dx, -dy). Because images have +y downward, using -dy aligns vertical up as reference. Resulting angle is signed: positive means tilt toward +x (to the right in image coordinates).
#Returns float degrees (can be negative when person leans backward/left etc).
#Note you also stored an earlier unsigned variant in other code; here you return signed tilt directly.

def torso_pelvis_twist_2d(sh_l, sh_r, hip_l, hip_r):
    if None in (sh_l, sh_r, hip_l, hip_r):
        return np.nan

    v_sh = np.array([sh_r[0] - sh_l[0], sh_r[1] - sh_l[1]])
    v_hip = np.array([hip_r[0] - hip_l[0], hip_r[1] - hip_l[1]])

    n1 = np.linalg.norm(v_sh)
    n2 = np.linalg.norm(v_hip)

    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan

    cos_ang = np.dot(v_sh, v_hip) / (n1 * n2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)

    # return float(np.degrees(np.arccos(cos_ang)))
    cross = np.cross(v_sh, v_hip)
    angle = np.degrees(np.arccos(cos_ang))
    return float(angle if cross > 0 else -angle)

# Simple EMA(Exponential Moving Average) smoother
class EMA:
    def __init__(self, alpha):
        self.alpha = alpha #Controls smoothness.
        self.v = None #Stores previous smoothed value.

    def update(self, x):
    #This updates EMA with new value x

        if x is None or (isinstance(x, float) and np.isnan(x)): #If invalid(Pose detection sometimes outputs: None, NaN, Missing keypoints): Do NOT update (This prevents signal breaking)
            return self.v #Return previous smoothed value
        
        if self.v is None:
        #For first data point: v0 ​= x0​ (because no history exists)
            self.v = float(x)

        else:
            #Mathematical func of EMA:
            self.v = float(self.alpha * x + (1 - self.alpha) * self.v)
            # α% comes from current frame
            # (1−α)% comes from previous smooth value
        return self.v
#Without smoothing:
# Angle: 10, 12, 8, 14, 9, 13 (Jittery)
#With EMA (α=0.3):
# Angle: 10, 10.6, 9.98, 11.8, 10.5, 11.6 (Much smoother)


#This Function Makes The Pipeline STABLE.
def nan_to_none(x):
#Instead of propagating NaN through system, you convert it to None.
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except Exception:
        pass
    return x


#It cleans a feature dictionary, adds a timestamp, and appends it as a new row to a CSV file (also, creating the header if the file doesn't exist).
def save_final_snapshot_csv(snapshot, fname="Final_Kick_Features.csv"):
    out = {k: ("" if v is None else v) for k,v in snapshot.items()}
    out["captured_at"] = datetime.now().isoformat()
    fieldnames = list(out.keys())
    write_header = not os.path.exists(fname)
    with open(fname, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(out)

#----------------------------------------------------------
# Middleware: Limit Upload Size Using Content-Length Header
#-----------------------------------------------------------
#This middleware checks the request's CONTENT-LENGTH[an HTTP header that tells the server how many bytes are in the request (or response) body] header and rejects the request with a 413 (Upload Too Large) error if it exceeds the allowed maximum body size.
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

#-------------
# FastAPI app
#-------------

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="KickCoach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MaxSizeMiddleware, max_body=MAX_UPLOAD_BYTES)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    #simple readiness: interpreter should be loaded
    return {"status": "ready", "model_loaded": True}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a video file (multipart form). The endpoint processes the video
    and returns the final kick snapshot captured at the first detected kick.
    """
    #basic MIME Check (Ensures user uploads only Video files.)
    ctype = (file.content_type or "").lower()
    if not ctype.startswith(ALLOWED_MIME_PREFIX):
        logger.warning("Rejecting file with content-type: %s", ctype)
        raise HTTPException(status_code=415, detail="Unsupported file type; expect video/*")

    #save uploaded file temporarily (streamed write)
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

        #Open Video with OpenCV:
        cap = cv2.VideoCapture(tmp.name)
        #This loads the video.
        if not cap.isOpened():
            logger.error("Cannot open video file %s", tmp.name)
            raise HTTPException(status_code=400, detail="Could not open uploaded video.")

         #FPS is needed because:
        # - Speed = pixels per frame × fps
        # - Angular velocity = angle difference × fps
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_ESTIMATE
        kick_cooldown_frames = max(1, int(round(KICK_COOLDOWN_SECS * fps))) #This converts seconds → number of frames.
        kick_cooldown = 0 #detection allowed

        #These smooth noisy values
        knee_ema = EMA(SMOOTH_ALPHA)
        trunk_ema = EMA(SMOOTH_ALPHA)
        hip_ema = EMA(SMOOTH_ALPHA)

        prev_ankle = None #Previous ankle position
        prev_knee_angle = None #Previous knee angle
        final_snapshot = None #First detected kick
        frame_idx = 0

        #Joint Indices:
        LHIP, RHIP = 11, 12 #Keypoint 11 = Left Hip, Keypoint 12 = Right Hip
        LKNEE, RKNEE = 13, 14 #Keypoint 13 = Left Knee, Keypoint 14 = Right Knee
        LANKLE, RANKLE = 15, 16 #Keypoint 15 = Left Ankle, Keypoint 16 = Right Ankle
        LSH, RSH = 5, 6 #Keypoint 5 = Left Shoulder, Keypoint 6 = Right Shoulder
        #These are MoveNet keypoint indices.

        #Process Frames:
        while True:
            ret, frame = cap.read() #Reads one frame at a time
            if not ret: #Stops when:
                break   #Video ends
            h, w, _ = frame.shape #height, width

            try:
                #Prepare Input For MoveNet (normalized frame) --> 192 x 192 x 3 (float32)
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

            #Kicking Leg (Chooses: right/left Leg indices)
            if KICKING_LEG.lower().startswith("r"):
                hip_idx, knee_idx, ankle_idx = RHIP, RKNEE, RANKLE
            else:
                hip_idx, knee_idx, ankle_idx = LHIP, LKNEE, LANKLE

            #Extract (x,y, confidence)
            def to_xy_conf(idx):
                y, x, c = pixel_kps[idx]
                return (float(x), float(y)), float(c)

            hip_pt, hip_conf = to_xy_conf(hip_idx)
            knee_pt, knee_conf = to_xy_conf(knee_idx)
            ankle_pt, ankle_conf = to_xy_conf(ankle_idx)

            #KNEE ANGLE: (computes angle at knee joint, Only if confidence is high enough)
            knee_angle = np.nan
            if min(hip_conf, knee_conf, ankle_conf) >= CONF_THRESH:
                knee_angle = angle_between_points(hip_pt, knee_pt, ankle_pt)

            #TRUNK VALUE: (measures body lean left/right)
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

            # #HIP ROTATION: (measures how open hip is relative to torso, if Camera setup is consistent)
            # hip_rotation = np.nan
            # if min(sh_l_conf, sh_r_conf, hip_conf, knee_conf) >= CONF_THRESH:
            #     sh_mid = (
            #         (sh_l_pt[0] + sh_r_pt[0]) / 2.0,
            #         (sh_l_pt[1] + sh_r_pt[1]) / 2.0
            #     )
            #     hip_rotation = angle_between_points(sh_mid, hip_pt, knee_pt)
            # HIP ROTATION (torso vs pelvis twist proxy)
            hip_rotation = np.nan

            try:
                hip_l_pt, hip_l_conf = to_xy_conf(LHIP)
                hip_r_pt, hip_r_conf = to_xy_conf(RHIP)

                if min(sh_l_conf, sh_r_conf, hip_l_conf, hip_r_conf) >= CONF_THRESH:
                    hip_rotation = torso_pelvis_twist_2d(
                        sh_l_pt, sh_r_pt,
                        hip_l_pt, hip_r_pt
                    )
            except Exception:
                hip_rotation = np.nan

            #ANKLE SPEED:
            ankle_speed_pps = np.nan
            if prev_ankle is not None and ankle_conf >= CONF_THRESH:
                dx = ankle_pt[0] - prev_ankle[0]
                dy = ankle_pt[1] - prev_ankle[1]
                ankle_speed_pxpf = math.hypot(dx, dy)
                ankle_speed_pps = ankle_speed_pxpf * fps
            prev_ankle = ankle_pt

            #KNEE ANGULAR VELOCITY:
            knee_ang_vel_dps = np.nan
            if prev_knee_angle is not None and not np.isnan(knee_angle):
                knee_ang_vel_dps = (knee_angle - prev_knee_angle) * fps
            prev_knee_angle = knee_angle if not np.isnan(knee_angle) else prev_knee_angle

            #Apply EMA Smoothing:
            sm_knee = knee_ema.update(knee_angle)
            sm_trunk = trunk_ema.update(trunk_val)
            sm_hip = hip_ema.update(hip_rotation)
            #Reduces jitter.

            #KICK DETECTION LOGIC:
            #Your system detects a kick when: The foot is moving very fast AND the knee is snapping quickly at the same time.
            #Then it:
            # -Saves that moment
            # -Prevents double detection
            # -Continues processing
            if kick_cooldown > 0:
                kick_cooldown -= 1

            # is_fast = isinstance(ankle_speed_pps, (int, float)) and not np.isnan(ankle_speed_pps) and ankle_speed_pps > KICK_THRESHOLD_PPS
            is_fast = (
                isinstance(ankle_speed_pps, (int, float)) and
                not np.isnan(ankle_speed_pps) and
                ankle_speed_pps > KICK_THRESHOLD_PPS #The foot is moving fast enough to possibly be a kick.
            )

            # is_explosive = isinstance(knee_ang_vel_dps, (int, float)) and not np.isnan(knee_ang_vel_dps) and abs(knee_ang_vel_dps) > KNEE_ANG_VEL_THRESHOLD
            is_explosive = (
                isinstance(knee_ang_vel_dps, (int, float)) and
                not np.isnan(knee_ang_vel_dps) and
                abs(knee_ang_vel_dps) > KNEE_ANG_VEL_THRESHOLD #How fast the knee angle is changing (degrees per second) --> If this is high: The leg is extending explosively.
            )

            if (is_fast and is_explosive and kick_cooldown == 0 and final_snapshot is None):
            #All of these must be TRUE:
            #1. Foot is moving fast
            #2. Knee is extending explosively
            #3. Cooldown is zero (not recently detected)
            #4. No kick has already been saved
            # Only then: You declare a KICK DETECTED.
            
                final_snapshot = {
                    "frame_number": int(frame_idx),
                    "knee_angle": nan_to_none(sm_knee),
                    "trunk": nan_to_none(sm_trunk),
                    "hip_rotation": nan_to_none(sm_hip),
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
