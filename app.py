import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from collections import deque
import threading
import time
import json
import asyncio
import atexit
import logging
import subprocess
import tempfile
import traceback
from bleak import BleakScanner, BleakClient
from emotion_detector import (
    LiveEmotionDetector, list_input_devices, _read_wav_16k, EMOTIONS,
    AUDIO_SAMPLE_RATE, AUDIO_WINDOW_SECONDS, UPLOAD_HOP_SECONDS,
    _make_vad, _dbfs, VAD_CHUNK_SAMPLES, VAD_MIN_SPEECH_SECONDS,
)

# ============================================================
# NO TENSORFLOW / NO DEEPFACE VERSION
# Python 3.12 friendly version using OpenCV Haar face + smile
# Robot emotion actions are preserved using BLE + taught poses.
# ============================================================

app = Flask(__name__)

# ============================================================
# CAMERA GLOBALS
# ============================================================
latest_frame = None
camera_running = True
detection_active = True
history = deque(maxlen=5)   # face streak buffer (stabilisation / debounce)
lock = threading.Lock()     # guards history + every emotion state dict below

# ============================================================
# CAMERA STATUS - what the UI shows instead of a silent black box
# ------------------------------------------------------------
# state is one of: starting | connected | denied | error | stopped
# Written only by _open_capture()/update_camera() (the camera thread);
# read by /status. "denied" is a best-effort guess (OpenCV gives no reliable
# permission-denied signal), reserved for webcam source 0 failing to open.
# ============================================================
camera_status_lock = threading.Lock()
camera_status = {"state": "starting", "detail": "Opening camera...", "source": "webcam"}


def _set_camera_status(state, detail, source):
    with camera_status_lock:
        camera_status.update({"state": state, "detail": detail, "source": source})
    print(f"[CAMERA] status -> {state}: {detail}")


def _camera_status_snapshot():
    """Consistent copy - the camera thread writes these fields concurrently."""
    with camera_status_lock:
        return dict(camera_status)

# ============================================================
# EMOTION STATE - SINGLE SOURCE OF TRUTH
# ------------------------------------------------------------
# Three layers are kept strictly separate and never overwrite each other:
#
#   FACE   raw webcam / test-video prediction (debounced by the streak lock)
#   VOICE  stable tone-of-voice prediction from emotion_detector.py
#   MANUAL explicit click on a "Manual Emotion Trigger" button
#   FINAL  the fused result. Music, background, robot action, pose key and
#          every "final" UI field read FINAL and nothing else.
#
# Fusion rule (priority, highest first):  MANUAL > VOICE > FACE
#   - VOICE only competes while the microphone is running and its last stable
#     result is newer than VOICE_EMOTION_TTL seconds.
#   - MANUAL only competes for MANUAL_EMOTION_TTL seconds after the click.
#   - FACE is the fallback and never expires.
# Consequence: while voice is active, a face result can NEVER replace FINAL.
# ============================================================
EMOTION_SOURCE_PRIORITY = ("manual", "voice", "face")
# When no sensor has produced anything yet, FINAL falls back to this. It is
# reported with source "default": the UI shows it, but it never moves the robot
# and never starts music, because nothing has actually been detected.
DEFAULT_FINAL_EMOTION = "neutral"
VOICE_EMOTION_TTL = 20.0     # seconds a stable voice result stays authoritative
MANUAL_EMOTION_TTL = 30.0    # seconds a manual button click stays authoritative
FACE_EMOTION_TTL = None      # face never expires on its own
# The FACE layer locks onto the first 5 identical frames and otherwise never runs
# the cascade again (see generate_frames), so a webcam session or an uploaded test
# video would get stuck on whatever expression appeared in its first second forever.
# Re-arming it periodically lets it keep tracking a face whose expression changes,
# which is the whole point of the video-upload emotion test feature.
FACE_RELOCK_SECONDS = 4.0

face_state = {"emotion": None, "confidence": 0.0, "locked": False, "seq": 0, "ts": 0.0}
# voice_state holds the STABLE (debounced) voice emotion - the only one that
# takes part in fusion. voice_raw_state holds the latest per-window reading and
# is for display only, so raw and stable are never confused with each other.
#
# "kind" distinguishes the two things that can occupy this slot:
#   "live"   - the live microphone's debounced stable emotion (emotion_detector.py)
#   "upload" - the completed one-shot analysis of an uploaded file's own audio
# Both compete in fusion identically (manual > voice > face), but their
# freshness rules differ: "live" additionally requires the mic to still be
# running AND uses VOICE_EMOTION_TTL, since it is a continuously-refreshed
# reading. "upload" describes a whole file rather than a moment, so it is
# exempt from the TTL (see _recompute_final_locked) and instead stays pinned
# until the video source changes (reset_detection_state / video_use clear it
# explicitly).
voice_state = {"emotion": None, "confidence": 0.0, "active": False, "kind": None, "seq": 0, "ts": 0.0}
voice_raw_state = {"emotion": None, "confidence": 0.0, "streak": 0, "streak_required": 3,
                   "speech_detected": False}
manual_state = {"emotion": None, "confidence": 1.0, "seq": 0, "ts": 0.0}
final_state = {"emotion": None, "source": None, "confidence": 0.0, "ts": 0.0}

SOURCE_STATES = {"face": face_state, "voice": voice_state, "manual": manual_state}
SOURCE_TTL = {"face": FACE_EMOTION_TTL, "voice": VOICE_EMOTION_TTL, "manual": MANUAL_EMOTION_TTL}

# Last robot action taken for a FINAL emotion (shown in the UI).
last_final_action = None

_emotion_seq = 0

# ============================================================
# FUSED-STATE SEQUENCE NUMBER
# ------------------------------------------------------------
# Bumped by _recompute_final_locked() whenever the *visible* fused result
# (emotion, source or confidence) actually changes. Every response that
# carries the fused emotion is stamped with it, so the browser can order
# responses that arrive out of sequence from DIFFERENT endpoints
# (/status and /voice/status are polled by separate loops) and refuse to
# render an older snapshot over a newer one. Without this, the camera card
# and the voice card each rendered their own independently-fetched copy of
# the fused emotion and could legitimately disagree on screen.
# ============================================================
_final_state_seq = 0
_final_state_changed_at = 0.0

# ============================================================
# TEST VIDEO UPLOAD (feed pre-recorded clips instead of the webcam)
# ============================================================
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Only these can be fed to cv2.VideoCapture as a *camera source* (they have a
# picture track). Audio-only uploads are analysed but never become the camera
# source - handing a .mp3 to cv2.VideoCapture just fails and blanks the feed.
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
# Audio-only uploads. ffmpeg decodes all of these, so the acoustic pipeline is
# identical for every one of them (see _extract_audio_wav).
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma"}
ALLOWED_MEDIA_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS

# Reject oversized uploads with a clean JSON error instead of a stack trace.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

video_source_lock = threading.Lock()
current_video_source = 0  # 0 = live webcam, otherwise a path to an uploaded video

# ============================================================
# UPLOADED VIDEO AUDIO EMOTION ANALYSIS
# ------------------------------------------------------------
# The FACE layer (above) reads an uploaded file's video frames continuously,
# same as the webcam. This is a separate, one-shot pipeline that reads the
# same file's own audio track and runs it through the SAME acoustic model
# used for the live microphone (the multi-dataset HuBERT ensemble from ser/,
# or the single pretrained superb/hubert-base-superb-er when no ensemble has
# been built - either way loaded exactly once in
# emotion_detector.LiveEmotionDetector, never a second model instance).
# Its result is shown in its own UI panel AND fed into SOURCE_STATES via
# apply_voice_emotion(kind="upload"), so it competes in the manual>voice>face
# fusion exactly like the live microphone (see the "kind" note on voice_state
# above) and can trigger the robot. It is superseded the normal way by a newer
# voice/manual result, and is explicitly cleared when the video source changes
# (reset_detection_state / video_use) rather than on a timer.
# ============================================================
upload_analysis_lock = threading.Lock()
upload_analysis_state = {
    "filename": None,
    "request_id": None,
    "kind": None,              # video | audio
    "status": "idle",          # idle | processing | complete | no_speech | error
    "emotion": None,
    "confidence": None,
    "probabilities": {},
    # Per-member breakdown of THIS file's result - the uploaded-file twin of the
    # live path's member_probabilities, and the thing to read when the ensemble's
    # answer for a clip is surprising.
    "member_probabilities": {},
    "member_votes": {},        # per member: how many windows it called each emotion
    "member_dissenting": [],   # members whose own average disagrees with the ensemble
    "combiner": None,
    "model": None,
    "windows_analyzed": 0,     # windows that actually contained speech and were classified
    "windows_total": 0,        # windows examined, including the ones skipped as non-speech
    "duration": None,
    "error": None,
    "message": None,
    "started_at": None,
    "finished_at": None,
    "processing_time": None,
}

# The uploaded-file pipeline MUST gate on speech exactly like the live
# microphone does. The acoustic model is a 4-way classifier with no "no speech"
# class, so it answers confidently for anything it is given: measured on this
# machine, pure digital silence -> ANGRY 46 %, and quiet room tone -> HAPPY 72 %.
# Feeding it un-gated windows is what made an uploaded clip report a confident
# emotion that nothing in the audio supports. One VAD instance is built lazily
# and reused (never one per request); it is not thread-safe, so it is serialised.
_upload_vad = None
_upload_vad_lock = threading.Lock()

# Per-window upload logging is diagnostic only: the one-line RESULT summary is
# always printed, the window-by-window detail only when it is asked for.
#   UPLOAD_LOG_WINDOWS=1 python app.py
UPLOAD_LOG_WINDOWS = os.environ.get("UPLOAD_LOG_WINDOWS", "0") == "1"


def _speech_ratio(segment):
    """Fraction of the segment's 32 ms chunks that the VAD calls speech."""
    global _upload_vad
    with _upload_vad_lock:
        if _upload_vad is None:
            _upload_vad, warning = _make_vad(os.environ.get("VOICE_VAD", "silero"))
            if warning:
                print(f"[UPLOAD] {warning}")
            print(f"[UPLOAD] Voice activity detection backend: {_upload_vad.name}")
        n = len(segment) // VAD_CHUNK_SAMPLES
        if n == 0:
            return 0.0
        _upload_vad.reset()
        speech = 0
        for i in range(n):
            chunk = segment[i * VAD_CHUNK_SAMPLES:(i + 1) * VAD_CHUNK_SAMPLES]
            try:
                if _upload_vad.is_speech(chunk):
                    speech += 1
            except Exception as exc:
                print(f"[UPLOAD] VAD chunk failed ({exc}); treating as non-speech")
        return speech / float(n)
_upload_analysis_seq = 0  # guards against an old file's slow analysis overwriting a newer one's


def _extension(filename):
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def _upload_analysis_snapshot():
    with upload_analysis_lock:
        return dict(upload_analysis_state)


def allowed_video_file(filename):
    """True only for files that can become the camera source (have a picture track)."""
    return _extension(filename) in ALLOWED_VIDEO_EXTENSIONS


def allowed_audio_file(filename):
    return _extension(filename) in ALLOWED_AUDIO_EXTENSIONS


def allowed_media_file(filename):
    """True for anything we accept for upload: video OR audio."""
    return _extension(filename) in ALLOWED_MEDIA_EXTENSIONS


def media_kind(filename):
    return "video" if allowed_video_file(filename) else (
        "audio" if allowed_audio_file(filename) else "unknown")

# OpenCV built-in Haar cascades. No TensorFlow required.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# Existing music mapping retained. Entries whose file is missing are dropped at
# start-up: pointing the player at a URL that 404s made it fail silently, which
# looked exactly like "the audio stopped working" with nothing in the log.
_SONG_CANDIDATES = {
    "happy": "/static/music/happy.m4a",
    "sad": "/static/music/sad.m4a",
    "angry": "/static/music/angry.m4a",
    "fear": "/static/music/fear.m4a",
    "surprise": "/static/music/surprise.m4a",
    "disgust": "/static/music/disgust.m4a",
    "neutral": "/static/music/neutral.m4a",
    "contempt": "/static/music/contempt.m4a",
}
_STATIC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
song_links = {
    emotion: url for emotion, url in _SONG_CANDIDATES.items()
    if os.path.isfile(os.path.join(_STATIC_ROOT, url.replace("/static/", "", 1)))
}
_missing_songs = sorted(set(_SONG_CANDIDATES) - set(song_links))
if _missing_songs:
    print(f"[MUSIC] No audio file for: {', '.join(_missing_songs)} "
          f"(those emotions will simply play nothing)")

# UI background colors can consume this from /status if required.
background_colors = {
    "happy": "#fff176",
    "angry": "#ef5350",
    "neutral": "#90caf9",
    "sad": "#5c6bc0",
    "fear": "#b39ddb",
    "surprise": "#ffb74d",
    "disgust": "#81c784",
    "contempt": "#b0bec5",
}

# ============================================================
# LIVE VOICE EMOTION SETTINGS
# Emotion comes from the tone of voice only (emotion_detector.py).
# No speech-to-text is used, so the words spoken do not matter.
# ============================================================
# Keep False while testing: stable voice emotions are only logged as
# "[ROBOT SIMULATION] Would trigger ...". Set True to run the real robot action.
VOICE_ROBOT_TRIGGER_ENABLED = False

# Microphone: None = system default, or an index / name substring listed by
# GET /voice/devices. Can also be chosen in the web page, or set with the
# VOICE_INPUT_DEVICE environment variable.
VOICE_INPUT_DEVICE = os.environ.get("VOICE_INPUT_DEVICE") or None

# Load the emotion model in the background at start-up so "Start Listening" is quick.
VOICE_PRELOAD_MODEL = True

# ============================================================
# ROBOT TEACH POSES
# ============================================================
POSE_FILE = "xarm_emotion_poses.json"

BASE_CENTER = 500
GRIPPER_OPEN = 240
GRIPPER_CLOSE = 150
GRIPPER_CLOSE_FIRM = 760

MOVE_FAST = 700
MOVE_NORMAL = 1000
MOVE_SLOW = 1500

# Emotion to taught pose key mapping.
EMOTION_TO_POSE = {
    "happy": "happy_30_right",
    "angry": "angry_60_right",
    "neutral": "neutral_30_left",
    "sad": "sad_60_left",
}

# Default placeholders. Replace by using /robot/save_pose or editing JSON.
DEFAULT_POSES = {
    "happy_30_right": {"1": GRIPPER_OPEN, "2": 610, "3": 720, "4": 500, "5": 360, "6": 625},
    "angry_60_right": {"1": GRIPPER_OPEN, "2": 610, "3": 720, "4": 500, "5": 360, "6": 750},
    "neutral_30_left": {"1": GRIPPER_OPEN, "2": 610, "3": 720, "4": 500, "5": 360, "6": 375},
    "sad_60_left": {"1": GRIPPER_OPEN, "2": 610, "3": 720, "4": 500, "5": 360, "6": 250},
    "front_safe": {"1": GRIPPER_OPEN, "2": 500, "3": 500, "4": 500, "5": 500, "6": BASE_CENTER},
    "front_present": {"1": GRIPPER_CLOSE_FIRM, "2": 520, "3": 600, "4": 500, "5": 520, "6": BASE_CENTER},
    "handover_forward": {"1": GRIPPER_CLOSE_FIRM, "2": 450, "3": 650, "4": 500, "5": 450, "6": BASE_CENTER} # Added placeholder
}


def load_robot_poses():
    if not os.path.exists(POSE_FILE):
        with open(POSE_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_POSES, f, indent=4)
        return dict(DEFAULT_POSES)

    try:
        with open(POSE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in DEFAULT_POSES.items():
            data.setdefault(key, value)
        return data
    except Exception:
        return dict(DEFAULT_POSES)


def save_robot_poses(poses):
    with open(POSE_FILE, "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=4)


robot_poses = load_robot_poses()

# ============================================================
# BLE LOOP + ROBOT CLASS
# ============================================================
class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class XArmBLE:
    SERVO_MOVE_TIME_WRITE = 0x03
    SERVO_MOVE_STOP = 0x14

    def __init__(self):
        self.client = None
        self.connected = False
        self.writable_chars = []
        self.write_char = None
        self.write_handle = None
        self.last_device = None
        self.last_write_index = 0
        self.busy = False
        self.last_emotion_executed = None
        self.last_action_time = 0

    async def scan(self, timeout=10):
        devices_out = []
        try:
            found = await BleakScanner.discover(timeout=timeout, return_adv=True)
            for address, data in found.items():
                device = data[0]
                adv = data[1]
                devices_out.append({
                    "name": device.name if device.name else "Unknown",
                    "address": device.address,
                    "rssi": getattr(adv, "rssi", "NA")
                })
        except TypeError:
            devices = await BleakScanner.discover(timeout=timeout)
            for device in devices:
                devices_out.append({
                    "name": device.name if device.name else "Unknown",
                    "address": device.address,
                    "rssi": "NA"
                })
        return devices_out

    async def connect(self, address, write_index=0):
        if self.connected:
            return True, "Robot already connected"

        self.client = BleakClient(address)
        await self.client.connect()

        if not self.client.is_connected:
            return False, "BLE connection failed"

        self.writable_chars = []
        print("\n========== BLE SERVICES ==========")
        for service in self.client.services:
            print("Service:", service.uuid)
            for char in service.characteristics:
                print("  UUID      :", char.uuid)
                print("  Handle    :", char.handle)
                print("  Properties:", char.properties)
                if "write" in char.properties or "write-without-response" in char.properties:
                    self.writable_chars.append(char)

        print("\n========== WRITABLE CHARACTERISTICS ==========")
        for i, char in enumerate(self.writable_chars):
            print(f"[{i}] UUID={char.uuid}, HANDLE={char.handle}, PROPS={char.properties}")
        print("==============================================\n")

        if not self.writable_chars:
            await self.client.disconnect()
            return False, "No writable BLE characteristic found"

        if write_index >= len(self.writable_chars):
            write_index = 0

        self.write_char = self.writable_chars[write_index]
        self.write_handle = self.write_char.handle
        self.connected = True
        self.last_device = address
        self.last_write_index = write_index
        return True, f"Connected. Write Index={write_index}, Handle={self.write_handle}"

    async def disconnect(self):
        try:
            if self.client and self.client.is_connected:
                await self.client.disconnect()
        except Exception:
            pass

        self.client = None
        self.connected = False
        self.write_char = None
        self.write_handle = None
        self.writable_chars = []
        self.busy = False
        return True

    def build_packet(self, command, params=None):
        if params is None:
            params = []
        length = len(params) + 2
        packet = bytearray([0x55, 0x55, length, command])
        for p in params:
            packet.append(int(p) & 0xFF)
        return packet

    async def send_packet(self, command, params=None):
        if not self.connected:
            raise RuntimeError("Robot not connected")
        if not self.client or not self.client.is_connected:
            raise RuntimeError("BLE client disconnected")
        packet = self.build_packet(command, params)
        print("Robot packet:", packet.hex(" "), "handle:", self.write_handle)
        if self.write_handle is None:
            raise RuntimeError("Write handle not set")
        await self.client.write_gatt_char(self.write_handle, packet, response=False)
        
        

    async def move_servos(self, servo_positions, move_time=1000):
        params = []
        servo_count = len(servo_positions)
        time_low = move_time & 0xFF
        time_high = (move_time >> 8) & 0xFF
        params.append(servo_count)
        params.append(time_low)
        params.append(time_high)

        for servo_id, position in servo_positions.items():
            position = max(0, min(1000, int(position)))
            params.append(int(servo_id))
            params.append(position & 0xFF)
            params.append((position >> 8) & 0xFF)

        await self.send_packet(self.SERVO_MOVE_TIME_WRITE, params)

    async def stop(self):
        await self.send_packet(self.SERVO_MOVE_STOP, [])
        self.busy = False


robot_loop = AsyncLoopThread()
robot = XArmBLE()

# ============================================================
# CAMERA THREAD
# ============================================================
def _open_capture(source):
    label = "webcam(0)" if source == 0 else source
    source_name = "webcam" if source == 0 else os.path.basename(str(source))
    print(f"[CAMERA] Initializing camera index: {label}")
    _set_camera_status("starting", f"Opening {source_name} ...", source_name)
    cap = cv2.VideoCapture(source)
    opened = cap.isOpened()
    print(f"[CAMERA] isOpened(): {opened}")
    if opened:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[VIDEO] FPS = {fps}")
        print(f"[VIDEO] Frame count = {frame_count}")
        print(f"[VIDEO] Resolution = {int(width)}x{int(height)}")

        first_ok, _ = cap.read()
        print(f"[CAMERA] First frame read: {first_ok}")
        if first_ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) if source != 0 else None
            print("[CAMERA] Camera started successfully")
            _set_camera_status("connected", f"{source_name} streaming", source_name)
        else:
            print("[CAMERA ERROR] Could not read a frame after opening")
            _set_camera_status("error", f"Opened {source_name} but could not read a frame", source_name)
    else:
        print("[CAMERA ERROR] Could not open camera")
        if source == 0:
            print("[CAMERA ERROR] Check macOS camera permissions "
                  "(System Settings > Privacy & Security > Camera)")
            _set_camera_status(
                "denied",
                "Camera permission denied. Enable camera access for the application "
                "running Python/OpenCV.",
                source_name,
            )
        else:
            print(f"[CAMERA ERROR] Path = {label}")
            _set_camera_status("error", f"Could not open video file: {source_name}", source_name)
    return cap


def update_camera():
    global latest_frame
    source = 0
    cap = _open_capture(source)
    frame_counter = 0
    consecutive_failures = 0

    while camera_running:
        with video_source_lock:
            desired_source = current_video_source

        if desired_source != source:
            cap.release()
            source = desired_source
            cap = _open_capture(source)
            frame_counter = 0
            consecutive_failures = 0

        success, frame = cap.read()

        if (not success or frame is None) and source != 0:
            # Uploaded file reached the end: loop it so testing can continue.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()

        latest_frame = frame if success and frame is not None else None

        cam_snap = _camera_status_snapshot()
        if success and frame is not None:
            consecutive_failures = 0
            if cam_snap["state"] != "connected":
                _set_camera_status("connected", f"{cam_snap['source']} streaming", cam_snap["source"])
        else:
            consecutive_failures += 1
            # A handful of failed reads in a row (not just one dropped frame) means
            # the device really did go away mid-session - surface that instead of
            # silently freezing on the last good frame forever.
            if consecutive_failures == 100 and cam_snap["state"] == "connected":
                print("[CAMERA ERROR] Lost the video stream (no frames for ~1s)")
                _set_camera_status("error", "Lost the video stream", cam_snap["source"])

        frame_counter += 1
        if frame_counter % 60 == 1:
            print(f"[DETECTION] Processing frame #{frame_counter} (source={'webcam' if source == 0 else os.path.basename(str(source))}, ok={success and frame is not None})")

        time.sleep(0.01)

    cap.release()


cam_thread = threading.Thread(target=update_camera, daemon=True)
cam_thread.start()

# ============================================================
# SIMPLE OPENCV EMOTION ENGINE
# ============================================================
def simple_opencv_emotion(face_gray):
    """
    Python 3.12 friendly fallback.
    Detects happy using smile cascade, otherwise neutral.
    Angry/sad are supported by manual web trigger buttons/API.
    """
    smiles = smile_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.7,
        minNeighbors=18,
        minSize=(25, 25)
    )
    if len(smiles) > 0:
        return "happy"
    return "neutral"


def lock_emotion_and_trigger(emotion):
    """Manual button press: highest-priority source, then normal fusion."""
    return apply_manual_emotion(emotion)

# ============================================================
# ROBOT SEQUENCE
# ============================================================
def pose_to_int_dict(pose):
    return {int(k): int(v) for k, v in pose.items()}


def wait_motion(move_time):
    time.sleep(move_time / 1000.0 + 0.35)


def send_robot_pose_sync(pose, move_time):
    future = robot_loop.run_async(robot.move_servos(pose_to_int_dict(pose), move_time))
    future.result(timeout=10)


def execute_robot_pick_for_emotion(emotion):
    emotion = emotion.lower().strip()

    if emotion not in EMOTION_TO_POSE:
        print(f"Robot: emotion '{emotion}' is not mapped to bottle action")
        return

    now = time.time()
    if robot.busy:
        print("Robot busy; skipping new action")
        return

    if robot.last_emotion_executed == emotion and (now - robot.last_action_time) < 20:
        print("Robot: same emotion recently executed; skipping duplicate")
        return

    if not robot.connected:
        print("Robot is not connected. Emotion action skipped.")
        return

    robot.busy = True
    robot.last_emotion_executed = emotion
    robot.last_action_time = now

    try:
        pose_key = EMOTION_TO_POSE[emotion]
        pick_pose = dict(robot_poses[pose_key])
        front_safe = dict(robot_poses.get("front_safe", DEFAULT_POSES["front_safe"]))
        # Ensure 'handover_forward' exists or provide a fallback
        handover_forward = dict(robot_poses.get("handover_forward", DEFAULT_POSES["handover_forward"]))

        # Step 1: Front safe open
        front_safe["1"] = GRIPPER_OPEN
        send_robot_pose_sync(front_safe, MOVE_SLOW)
        wait_motion(MOVE_SLOW)

        # Step 2: Base rotation only
        rotate_pose = dict(front_safe)
        rotate_pose["6"] = pick_pose["6"]
        send_robot_pose_sync(rotate_pose, MOVE_SLOW)
        wait_motion(MOVE_SLOW)

        # Step 3: Approach above taught pick pose
        approach_pose = dict(pick_pose)
        approach_pose["1"] = GRIPPER_OPEN
        approach_pose["2"] = max(0, int(approach_pose["2"]) - 25)
        approach_pose["3"] = max(0, int(approach_pose["3"]) - 25)
        send_robot_pose_sync(approach_pose, MOVE_SLOW)
        wait_motion(MOVE_SLOW)

        # Step 4: Exact taught pick pose
        pick_pose["1"] = GRIPPER_OPEN
        send_robot_pose_sync(pick_pose, MOVE_SLOW)
        wait_motion(MOVE_SLOW)

        # Step 5: Close gripper
        send_robot_pose_sync({"1": GRIPPER_CLOSE}, MOVE_FAST)
        wait_motion(MOVE_FAST)
        send_robot_pose_sync({"1": GRIPPER_CLOSE_FIRM}, MOVE_FAST)
        wait_motion(MOVE_FAST)

        # Step 6: Lift bottle
        lift_pose = dict(pick_pose)
        lift_pose["1"] = GRIPPER_CLOSE_FIRM
        lift_pose["2"] = int(lift_pose["2"]) - 80
        lift_pose["3"] = int(lift_pose["3"]) + 80
        lift_pose["5"] = int(lift_pose["5"]) + 50
        send_robot_pose_sync(lift_pose, MOVE_SLOW)
        wait_motion(MOVE_SLOW)

        # Step 7: Move toward person using taught handover pose
        send_robot_pose_sync(handover_forward, MOVE_SLOW)
        wait_motion(MOVE_SLOW)

        # Step 8: Present bottle
        print("Waiting for user to take bottle...")
        time.sleep(5)

        # Step 9: Release bottle (handover)
        send_robot_pose_sync({"1": GRIPPER_OPEN}, MOVE_FAST)
        wait_motion(MOVE_FAST)

        # Step 10: Return to safe position
        send_robot_pose_sync(front_safe, MOVE_SLOW)
        wait_motion(MOVE_SLOW)

        print("Bottle handed over successfully")

    except Exception as e:
        print("Robot sequence error:", e)
    finally:
        robot.busy = False


def trigger_robot_for_emotion(emotion):
    t = threading.Thread(target=execute_robot_pick_for_emotion, args=(emotion,), daemon=True)
    t.start()

# ============================================================
# EMOTION FUSION ENGINE
# ------------------------------------------------------------
# Every emotion result in the app enters through apply_*_emotion(). Those are
# the ONLY writers of face_state / voice_state / manual_state, and
# _recompute_final_locked() is the ONLY writer of final_state. Nothing else may
# assign an emotion, which is what stops a late camera callback from silently
# replacing a newer voice result.
# ============================================================
def _recompute_final_locked():
    """Re-run the fusion rule. Caller MUST hold `lock`.

    Returns (changed, snapshot_of_final_state).
    """
    now = time.time()
    winner_source = None
    winner_state = None

    for source in EMOTION_SOURCE_PRIORITY:
        state = SOURCE_STATES[source]
        if not state["emotion"]:
            continue
        if source == "voice" and state.get("kind") != "upload" and not state["active"]:
            continue  # microphone stopped -> voice drops out of the fusion
        ttl = SOURCE_TTL[source]
        if source == "voice" and state.get("kind") == "upload":
            # An uploaded file's voice result describes THAT file, not a
            # moment in time, so it does not decay on the live-mic TTL - it
            # stays authoritative for as long as that file is selected and is
            # only dropped when the source changes (see reset_detection_state
            # / video_use), never on a timer.
            ttl = None
        if ttl is not None and (now - state["ts"]) > ttl:
            continue  # result too old to stay authoritative
        winner_source = source
        winner_state = state
        break

    previous = (final_state["emotion"], final_state["source"])
    previous_visible = (final_state["emotion"], final_state["source"],
                        round(float(final_state["confidence"]), 4))

    if winner_source is None:
        final_state.update({"emotion": DEFAULT_FINAL_EMOTION, "source": "default",
                            "confidence": 0.0, "ts": now})
    else:
        final_state.update({
            "emotion": winner_state["emotion"],
            "source": winner_source,
            "confidence": float(winner_state["confidence"]),
            "ts": now,
        })

    changed = previous != (final_state["emotion"], final_state["source"])

    # The robot must only be re-triggered when the emotion/source changes, but the
    # UI must re-render whenever the displayed confidence changes too, so the
    # sequence number tracks a slightly wider notion of "changed" than `changed`.
    global _final_state_seq, _final_state_changed_at
    visible_now = (final_state["emotion"], final_state["source"],
                   round(float(final_state["confidence"]), 4))
    if visible_now != previous_visible:
        _final_state_seq += 1
        _final_state_changed_at = now

    if changed:
        print(f"[FUSION] manual={(manual_state['emotion'] or '-').upper()} "
              f"voiceStable={(voice_state['emotion'] or '-').upper()} "
              f"(active={voice_state['active']}) "
              f"face={(face_state['emotion'] or '-').upper()} "
              f"(locked={face_state['locked']}) "
              f"-> final={(final_state['emotion'] or '-').upper()} "
              f"source={(final_state['source'] or '-').upper()} "
              f"confidence={final_state['confidence'] * 100:.0f}", flush=True)
    return changed, dict(final_state)


def _on_final_emotion_changed(snapshot):
    """FINAL changed: this is the one place that drives the robot. Never call with `lock` held."""
    global last_final_action
    emotion = snapshot["emotion"]
    source = snapshot["source"]

    if not emotion:
        print("[ROBOT] finalEmotion=None action=none (final emotion cleared)")
        with lock:
            last_final_action = None
        return

    if source == "default":
        # Nothing has been detected yet: show neutral, but do not move the arm.
        print("[ROBOT] finalEmotion=NEUTRAL action=none (default placeholder, "
              "no sensor has reported yet)")
        with lock:
            last_final_action = None
        return

    pose_key = EMOTION_TO_POSE.get(emotion, "not_mapped")
    if pose_key == "not_mapped":
        print(f"[ROBOT] emotion={emotion} action=none (emotion not mapped to a taught pose)")
        return

    # Voice-driven actions stay simulated until VOICE_ROBOT_TRIGGER_ENABLED is True,
    # exactly as before. Face/manual keep triggering the real sequence.
    if source == "voice" and not VOICE_ROBOT_TRIGGER_ENABLED:
        action = f"simulated {emotion}"
        print(f"[ROBOT SIMULATION] Would trigger {emotion}")
    else:
        if not robot.connected:
            note = " (robot not connected - action will be skipped)"
        elif robot.busy:
            note = " (robot busy - action will be skipped)"
        else:
            note = ""
        trigger_robot_for_emotion(emotion)
        action = f"robot requested for {emotion}{note}"

    print(f"[ROBOT] finalEmotion={emotion.upper()} source={source.upper()} "
          f"action={pose_key} detail={action}", flush=True)
    with lock:
        last_final_action = {
            "action": action,
            "emotion": emotion,
            "source": source,
            "pose_key": pose_key,
            "confidence": round(float(snapshot["confidence"]), 2),
            "time": time.time(),
        }


def _apply_emotion(source, emotion, confidence, extra=None, observed_at=None):
    """Record a new result for one source, then re-fuse.

    `observed_at` is when the observation was actually made (not when it was
    delivered). A result observed before the one already stored for the same
    source is dropped, so a slow/late callback can never move state backwards.
    """
    global _emotion_seq
    state = SOURCE_STATES[source]
    now = time.time()
    if observed_at is None:
        observed_at = now

    with lock:
        if observed_at < state["ts"]:
            print(f"[{source.upper()}] emotion={emotion} DROPPED (out of order: "
                  f"observed {state['ts'] - observed_at:.2f}s before the stored result)")
            return dict(final_state)

        _emotion_seq += 1
        state.update({
            "emotion": emotion,
            "confidence": float(confidence),
            "seq": _emotion_seq,
            "ts": observed_at,
        })
        if extra:
            state.update(extra)
        tag = {"face": "FACE LOCK", "voice": "VOICE STABLE->APP", "manual": "MANUAL"}[source]
        print(f"[{tag}] emotion={emotion.upper()} confidence={float(confidence) * 100:.0f}",
              flush=True)
        changed, snapshot = _recompute_final_locked()

    if changed:
        _on_final_emotion_changed(snapshot)
    return snapshot


def apply_face_emotion(emotion, confidence=1.0, locked=True, observed_at=None):
    """Raw camera / test-video result. Stays raw: fusion may ignore it, but it is never rewritten."""
    return _apply_emotion("face", emotion, confidence,
                          extra={"locked": bool(locked)}, observed_at=observed_at)


def apply_voice_emotion(emotion, confidence, observed_at=None, kind="live"):
    """Voice-layer result: "live" = stable tone-of-voice from emotion_detector.py
    (via the running microphone); "upload" = the completed one-shot analysis of
    an uploaded file's own audio (_analyze_uploaded_video_audio). Both compete
    in fusion identically; see the "kind" note on voice_state above for how the
    "still running" gate differs between them."""
    return _apply_emotion("voice", emotion, confidence,
                          extra={"active": True, "kind": kind}, observed_at=observed_at)


def apply_manual_emotion(emotion):
    """Manual emotion button: wins over voice and face for MANUAL_EMOTION_TTL seconds."""
    return _apply_emotion("manual", emotion, 1.0)


def refresh_final_emotion():
    """Re-run fusion for time-based changes only (TTL expiry, microphone stopped)."""
    with lock:
        changed, snapshot = _recompute_final_locked()
    if changed:
        _on_final_emotion_changed(snapshot)
    return snapshot


def emotion_snapshot():
    """Consistent copy of every emotion layer for the API/UI."""
    refresh_final_emotion()
    with lock:
        return {
            "face": dict(face_state),
            "voice": dict(voice_state),
            "voice_raw": dict(voice_raw_state),
            "manual": dict(manual_state),
            "final": dict(final_state),
            "last_action": dict(last_final_action) if last_final_action else None,
            "state_seq": _final_state_seq,
            "state_changed_at": _final_state_changed_at,
        }


def clear_emotion_source(source, reason=""):
    """Drop one source's result (e.g. face on a video-source switch) and re-fuse."""
    with lock:
        SOURCE_STATES[source].update({"emotion": None, "confidence": 0.0, "seq": 0, "ts": 0.0})
        if source == "face":
            face_state["locked"] = False
            history.clear()
        print(f"[{source.upper()}] cleared{(' - ' + reason) if reason else ''}")
        changed, snapshot = _recompute_final_locked()
    if changed:
        _on_final_emotion_changed(snapshot)
    return snapshot


def _maybe_relock_face():
    """Re-open the FACE layer once it has held its lock for FACE_RELOCK_SECONDS.

    Without this, a face that locks onto the first 5 identical frames (typically
    within the first second) never runs the cascade again for the rest of a
    webcam session or an uploaded test video, so an expression change later in
    the footage is never seen. Re-arming does not touch face_state["emotion"],
    so the last detected label keeps showing until a new streak locks in.
    """
    with lock:
        if not face_state["locked"] or (time.time() - face_state["ts"]) < FACE_RELOCK_SECONDS:
            return
        face_state["locked"] = False
        history.clear()
        print(f"[FACE] re-armed for re-detection (was locked to "
              f"{(face_state['emotion'] or '-').upper()} for {FACE_RELOCK_SECONDS:.0f}s+)")

# ============================================================
# VIDEO STREAM
# ============================================================
STREAM_TARGET_FPS = 20
STREAM_FRAME_INTERVAL = 1.0 / STREAM_TARGET_FPS

# Shared across generate_frames() (MJPEG push, used by browsers that support
# multipart/x-mixed-replace) AND video_frame() (single-JPEG poll, used by the
# page itself) so the "log every 30th frame" cadence in _render_detection_frame
# stays sane no matter which path is driving the camera right now.
_frame_log_counter = 0
_frame_log_lock = threading.Lock()


def _next_frame_log_index():
    global _frame_log_counter
    with _frame_log_lock:
        _frame_log_counter += 1
        return _frame_log_counter


def _render_and_encode_frame(quality=80):
    """Read latest_frame, run detection/overlay on a COPY of it, return JPEG
    bytes. The single place both video_feed (MJPEG) and video_frame (polled
    single image) get their picture from, so the two paths can never diverge."""
    frame = latest_frame

    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera Error: Cannot read webcam", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Check camera privacy settings", (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes()

    frame = frame.copy()
    try:
        _render_detection_frame(frame, _next_frame_log_index())
    except Exception as exc:
        # A per-frame detection error must never break the picture - that would
        # be exactly the silent "black box" the UI has no way to explain. Log it
        # loudly and keep showing a frame instead.
        print(f"[DETECTION ERROR] {exc}")
        traceback.print_exc()
        cv2.putText(frame, f"Detection error: {exc}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


def generate_frames():
    """MJPEG push stream (multipart/x-mixed-replace). Kept for browsers that
    support it (Chrome, Firefox), but NOT what the page itself uses any more -
    see video_frame() below and the ROOT CAUSE note there."""
    while True:
        loop_start = time.time()
        frame_bytes = _render_and_encode_frame()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        # Without this cap the generator re-reads/re-encodes as fast as the
        # webcam allows (measured ~135fps at 1920x1080, ~70MB/s) which floods
        # the browser's multipart/x-mixed-replace decoder and pegs a CPU core.
        elapsed = time.time() - loop_start
        remaining = STREAM_FRAME_INTERVAL - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _render_detection_frame(frame, face_log_counter):
    """Runs face detection/fusion overlay for one frame, in place on `frame`.

    Split out of generate_frames() so the whole thing can be wrapped in one
    try/except there without a deep nested indent change.
    """
    if not detection_active:
        cv2.putText(frame, "Detection Paused", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        face_should_detect = False
    else:
        _maybe_relock_face()
        face_should_detect = not face_state["locked"]

    if face_should_detect:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if face_log_counter % 30 == 1:
            print(f"[FACE] Faces detected = {len(faces)}")

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_gray = gray[y:y+h, x:x+w]
                if face_gray.size == 0:
                    # A face box that clips the frame edge (rare, but real - e.g. a
                    # face detected right at frame 0 of a freshly switched source)
                    # crops to an empty array; smile_cascade.detectMultiScale then
                    # raises cv2.error and used to kill the whole MJPEG generator,
                    # which is exactly the "stream freezes/black box" symptom.
                    print("[FACE] Skipped an empty face crop (box touched the frame edge)")
                    continue

                # Timestamp the observation itself so a slow classification
                # cannot be applied on top of a newer result later on.
                observed_at = time.time()
                detected = simple_opencv_emotion(face_gray)
                if face_log_counter % 30 == 1:
                    print(f"[FACE RAW] emotion={detected.upper()} "
                          f"streak={list(history)}", flush=True)

                with lock:
                    history.append(detected)
                    stable = (len(history) == history.maxlen and len(set(history)) == 1)

                if stable:
                    # Face stabilisation/debounce is preserved: the streak lock
                    # freezes the FACE layer only. It no longer owns the final
                    # emotion, so it can never overwrite a newer voice result.
                    print(f"[FACE LOCK] emotion={detected.upper()} "
                          f"(5 identical frames; locks the FACE layer only)", flush=True)
                    apply_face_emotion(detected, confidence=1.0, locked=True,
                                       observed_at=observed_at)

                label = face_state["emotion"] or detected
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    elif detection_active:
        label = face_state["emotion"] or "Locked"
        cv2.putText(frame, f"Face locked: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Always show the fused result on the frame so the overlay can never
    # disagree with the rest of the UI.
    final_label = final_state["emotion"]
    final_source = final_state["source"]
    overlay = f"Final: {final_label.upper()} ({final_source})" if final_label else "Final: -"
    cv2.putText(frame, overlay, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

# ============================================================
# EXISTING FLASK ROUTES
# ============================================================
@app.route("/")
def index():
    global detection_active
    with lock:
        detection_active = True
    # Reload the page with a clean camera/manual state, but keep the voice
    # layer: the microphone may still be running, and the detector only emits
    # again when the stable emotion *changes*, so clearing it here would leave
    # the final emotion stuck on an old face result.
    clear_emotion_source("face", "page loaded")
    clear_emotion_source("manual", "page loaded")
    on_voice_update(voice_detector.get_status())
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG push stream. Left in place for compatibility, but the page itself
    no longer uses this - see /video_frame."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_frame")
def video_frame():
    """One current JPEG frame (camera or uploaded video, with the detection
    overlay already drawn), fetched by the page on a JS interval instead of a
    server-push MJPEG stream.
    ------------------------------------------------------------------
    ROOT CAUSE this exists for: Safari/WebKit does not reliably render
    multipart/x-mixed-replace in an <img> - the server-side stream was
    verified correct with curl (real JPEG frames, correct boundaries,
    correct Content-Type), but nothing appeared in Safari's <img>. That is a
    long-standing WebKit gap, not a bug in this stream: Chrome and Firefox
    support multipart/x-mixed-replace, Safari does not render it at all in
    many versions. A single-JPEG-per-request endpoint, polled from JS and
    swapped in via an object URL, works identically in every browser because
    it is just an ordinary image fetch - nothing multipart-specific for
    Safari to fail to support.
    ------------------------------------------------------------------
    """
    frame_bytes = _render_and_encode_frame()
    resp = Response(frame_bytes, mimetype="image/jpeg")
    # Every poll must fetch a fresh frame - a cached one would freeze the
    # preview exactly like the bug this endpoint exists to fix.
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/video/upload", methods=["POST"])
def video_upload():
    if "video" not in request.files:
        print("[UPLOAD ERROR] Request had no 'video' file part")
        return jsonify({"ok": False, "error": "no file part 'video'"})

    file = request.files["video"]
    print(f"[UPLOAD] File received: {file.filename!r}")
    if file.filename == "":
        print("[UPLOAD ERROR] Empty filename")
        return jsonify({"ok": False, "error": "no file selected"})
    if not allowed_media_file(file.filename):
        print(f"[UPLOAD ERROR] Unsupported file type: {file.filename!r}")
        return jsonify({"ok": False,
                        "error": f"Unsupported file type '.{_extension(file.filename) or '?'}'. "
                                 f"Supported: {', '.join(sorted(ALLOWED_MEDIA_EXTENSIONS))}."})

    filename = secure_filename(file.filename)
    if not allowed_media_file(filename):
        # secure_filename() can strip a non-ASCII name down to something with no
        # usable extension left; fall back rather than saving an unreadable blob.
        filename = f"upload_{int(time.time())}.{_extension(file.filename)}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    print(f"[UPLOAD] File type: {os.path.splitext(filename)[1]}")
    try:
        file.save(path)
    except Exception as exc:
        print(f"[UPLOAD ERROR] Could not save file: {exc}")
        return jsonify({"ok": False, "error": f"could not save file: {exc}"})
    size = os.path.getsize(path)
    if size == 0:
        os.remove(path)
        print("[UPLOAD ERROR] Uploaded file is empty")
        return jsonify({"ok": False, "error": "The uploaded file is empty (0 bytes)."})
    kind = media_kind(filename)
    print(f"[UPLOAD] Saved to: {path} ({size} bytes, kind={kind})")
    return jsonify({"ok": True, "filename": filename, "kind": kind, "size": size})


@app.errorhandler(413)
def _upload_too_large(_exc):
    return jsonify({"ok": False,
                    "error": f"File is too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."}), 413


@app.route("/video/list")
def video_list():
    files = sorted(f for f in os.listdir(UPLOAD_FOLDER) if allowed_media_file(f))
    return jsonify({"ok": True,
                    "files": files,
                    "items": [{"filename": f, "kind": media_kind(f)} for f in files]})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve an uploaded clip's raw bytes so the browser can preview it with a
    native <video controls> element (Play/Pause/Seek/Mute/Volume all provided
    by the browser itself - nothing custom to keep in sync)."""
    filename = secure_filename(filename)
    if not allowed_media_file(filename):
        return jsonify({"ok": False, "error": "unsupported file type"}), 404
    if not os.path.isfile(os.path.join(UPLOAD_FOLDER, filename)):
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(UPLOAD_FOLDER, filename)


def _extract_audio_wav(video_path, out_wav_path):
    """Extract a 16 kHz mono PCM WAV from video_path using the ffmpeg binary
    bundled by imageio-ffmpeg (no system PATH/ffmpeg install required, so this
    works the same on the Windows target as it does here)."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(f"ffmpeg is not available ({exc}). "
                            f"Install it with: pip install imageio-ffmpeg")

    # -vn drops any picture track, so the SAME command handles .wav/.mp3/.m4a/
    # .mp4/.webm/... and every path reaches the model as 16 kHz mono float32.
    # The uploaded file is never assumed to already be WAV.
    cmd = [ffmpeg_exe, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
           "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "wav", out_wav_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out after 180 s - the file may be corrupt "
                           "or far too long.")

    if result.returncode != 0 or not os.path.isfile(out_wav_path) or os.path.getsize(out_wav_path) <= 44:
        stderr = result.stderr or ""
        lowered = stderr.lower()
        if "does not contain any stream" in lowered or "output file #0 does not contain" in lowered:
            raise RuntimeError("This file has no audio track, so there is no voice to analyse.")
        if "invalid data found" in lowered or "moov atom not found" in lowered:
            raise RuntimeError("The file could not be decoded - it looks corrupt or is not "
                               "really an audio/video file.")
        stderr_tail = "\n".join(stderr.strip().splitlines()[-6:])
        raise RuntimeError(f"Audio extraction failed (ffmpeg exit {result.returncode}): {stderr_tail}")


def _analyze_uploaded_video_audio(path, filename, request_id=None):
    """Background job: run the SAME acoustic emotion model used for the live
    microphone (via voice_detector, loaded once) against an uploaded file's own
    audio track, in sliding windows, and average the result.

    Speech gating is applied per window with the same VAD the live path uses:
    the model has no "not speech" class, so un-gated silence/music/noise is
    reported as a confident emotion that nothing in the audio supports.

    On success, feeds SOURCE_STATES via apply_voice_emotion(kind="upload"), so
    the result competes in the manual>voice>face fusion and can move the robot,
    same as the live microphone - see the "kind" note on voice_state.
    """
    global _upload_analysis_seq
    request_id = request_id or f"upl-{int(time.time() * 1000)}"
    started = time.time()
    with upload_analysis_lock:
        _upload_analysis_seq += 1
        my_seq = _upload_analysis_seq
        upload_analysis_state.update({
            "filename": filename, "request_id": request_id, "kind": media_kind(filename),
            "status": "processing", "emotion": None, "confidence": None,
            "probabilities": {}, "member_probabilities": {}, "member_votes": {},
            "member_dissenting": [], "combiner": None, "model": None,
            "windows_analyzed": 0, "windows_total": 0,
            "duration": None, "error": None, "message": None,
            "started_at": started, "finished_at": None, "processing_time": None,
        })
    print(f"[UPLOAD {request_id}] Processing started for {filename}", flush=True)

    def still_current():
        with upload_analysis_lock:
            return _upload_analysis_seq == my_seq

    def finish(**fields):
        """Publish a terminal result, but only if a newer analysis has not started."""
        with upload_analysis_lock:
            if _upload_analysis_seq != my_seq:
                print(f"[UPLOAD {request_id}] Superseded by a newer analysis - result discarded")
                return
            upload_analysis_state.update(dict(
                fields, finished_at=time.time(), processing_time=round(time.time() - started, 3)))

    tmp_wav = None
    try:
        if not os.path.isfile(path):
            raise RuntimeError("The uploaded file is no longer on disk.")

        tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="upload_audio_")
        os.close(tmp_fd)

        _extract_audio_wav(path, tmp_wav)
        audio = _read_wav_16k(tmp_wav)
        duration = len(audio) / float(AUDIO_SAMPLE_RATE)
        peak_db = _dbfs(audio)
        print(f"[UPLOAD {request_id}] Audio extracted: {duration:.1f}s at "
              f"{AUDIO_SAMPLE_RATE} Hz mono, level {peak_db:.1f} dBFS", flush=True)
        if duration < 1.0:
            raise RuntimeError(f"The audio is only {duration:.1f} s long - at least 1 s of "
                               f"speech is needed to detect an emotion.")

        if not still_current():
            return
        if not voice_detector._ensure_model():
            raise RuntimeError("The emotion model could not be loaded (see the server log).")

        win = int(AUDIO_WINDOW_SECONDS * AUDIO_SAMPLE_RATE)
        # Overlapping windows (3.0 s stepped by 1.5 s). The previous hop was a
        # full window, so an emotional peak that straddled a boundary was split
        # across two half-hearted windows and could lose to the calm speech on
        # either side of it. The live path has always overlapped; this makes the
        # uploaded-file path agree with it.
        hop = max(1, int(UPLOAD_HOP_SECONDS * AUDIO_SAMPLE_RATE))
        # A window must contain at least as much speech as the live path demands
        # before it is allowed to vote, so both paths apply the same standard.
        min_speech_ratio = VAD_MIN_SPEECH_SECONDS / AUDIO_WINDOW_SECONDS
        sums = {e: 0.0 for e in EMOTIONS}
        # Per-member running totals, so an uploaded file's result can be taken
        # apart the same way a live one can: which member argued for what.
        member_sums = {}
        member_votes = {}
        weight_total = 0.0
        windows = 0
        examined = 0
        combiner_kind = None

        for start_sample in range(0, len(audio), hop):
            if not still_current():
                return
            segment = audio[start_sample:start_sample + win]
            if len(segment) < AUDIO_SAMPLE_RATE:  # trailing sliver under 1 s
                continue
            examined += 1
            at = start_sample / float(AUDIO_SAMPLE_RATE)

            ratio = _speech_ratio(segment)
            if ratio < min_speech_ratio:
                if UPLOAD_LOG_WINDOWS:
                    print(f"[UPLOAD {request_id}] Window {examined} ({at:.1f}s) skipped: "
                          f"only {ratio * 100:.0f}% speech "
                          f"(needs {min_speech_ratio * 100:.0f}%), {_dbfs(segment):.1f} dBFS",
                          flush=True)
                continue

            detail = voice_detector._model.predict_detailed(segment)
            probs = detail["ensemble"]
            combiner_kind = detail.get("combiner")
            windows += 1
            # Windows are weighted by how much of them is actually speech, so a
            # window that only just cleared the gate counts for less than a full
            # one. With overlap this also stops the same half-second of audio
            # being counted twice at full strength.
            weight = float(ratio)
            weight_total += weight
            for e in EMOTIONS:
                sums[e] += probs[e] * weight
            for name, member in (detail.get("members") or {}).items():
                bucket = member_sums.setdefault(name, {e: 0.0 for e in EMOTIONS})
                for e in EMOTIONS:
                    bucket[e] += member[e] * weight
                top_member = max(member, key=member.get)
                member_votes.setdefault(name, {})
                member_votes[name][top_member] = member_votes[name].get(top_member, 0) + 1

            if UPLOAD_LOG_WINDOWS:
                top = max(probs, key=probs.get)
                line = (f"[UPLOAD {request_id}] Window {examined} ({at:.1f}s) "
                        f"speech={ratio * 100:.0f}% -> {top.upper()} ({probs[top] * 100:.0f}%)")
                members = detail.get("members") or {}
                if len(members) > 1:
                    line += "  | " + " ".join(
                        f"{name}={max(m, key=m.get)[:3].upper()}{max(m.values()) * 100:.0f}"
                        for name, m in members.items())
                    if detail.get("dissenting"):
                        line += f"  (dissent: {','.join(detail['dissenting'])})"
                print(line, flush=True)

        if not still_current():
            return

        if windows == 0:
            message = ("No speech was found in this file. The audio is silent, too quiet, "
                       "or contains only music/background noise, so no voice emotion can "
                       "be detected.")
            print(f"[UPLOAD {request_id}] {message}", flush=True)
            finish(status="no_speech", message=message, duration=round(duration, 2),
                   windows_total=examined, windows_analyzed=0)
            return

        divisor = weight_total or float(windows)
        averaged = {e: sums[e] / divisor for e in EMOTIONS}
        total = sum(averaged.values()) or 1.0
        averaged = {e: p / total for e, p in averaged.items()}
        emotion = max(averaged, key=averaged.get)
        confidence = averaged[emotion]

        member_averaged = {}
        for name, bucket in member_sums.items():
            member_total = sum(bucket.values()) or 1.0
            member_averaged[name] = {e: round(bucket[e] / member_total, 4) for e in EMOTIONS}
        dissenting = sorted(name for name, probs in member_averaged.items()
                            if max(probs, key=probs.get) != emotion)

        model_label = getattr(voice_detector._model, "name", voice_detector.model_name)
        print(f"[UPLOAD {request_id}] RESULT file={filename} duration={duration:.1f}s "
              f"speech_windows={windows}/{examined} emotion={emotion.upper()} "
              f"confidence={confidence * 100:.0f}% model={model_label} "
              f"took={time.time() - started:.1f}s", flush=True)
        if len(member_averaged) > 1:
            for name, probs in member_averaged.items():
                top = max(probs, key=probs.get)
                votes = member_votes.get(name, {})
                mark = "  <-- disagrees with the ensemble" if name in dissenting else ""
                print(f"[UPLOAD {request_id}]   {name:<22} {top.upper():<8}"
                      f"{probs[top] * 100:3.0f}%  window votes "
                      f"{ {k: v for k, v in sorted(votes.items())} }{mark}", flush=True)

        finish(status="complete", emotion=emotion, confidence=round(confidence, 4),
               probabilities={e: round(p, 4) for e, p in averaged.items()},
               member_probabilities=member_averaged, member_votes=member_votes,
               member_dissenting=dissenting, combiner=combiner_kind, model=model_label,
               windows_analyzed=windows, windows_total=examined,
               duration=round(duration, 2))

        # Feed the result into the SAME fusion the live microphone uses
        # (manual > voice > face, same VOICE_EMOTION_TTL, same
        # VOICE_ROBOT_TRIGGER_ENABLED simulation gate) - but only if a newer
        # analysis has not since started; still_current() guards this exactly
        # like finish() does, so a slow, now-superseded analysis can never
        # apply its result on top of whatever a newer one already published.
        if still_current():
            apply_voice_emotion(emotion, confidence, kind="upload")

    except Exception as exc:
        print(f"[UPLOAD {request_id} ERROR] {exc}", flush=True)
        finish(status="error", error=str(exc))
    finally:
        if tmp_wav and os.path.isfile(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


def _start_upload_analysis(path, filename):
    """Start (or restart) the one-shot audio analysis for a file.

    Exactly one analysis is ever in flight: _upload_analysis_seq makes any
    older worker discard its own result, so a slow analysis of a previous file
    can never publish over a newer one.
    """
    request_id = f"upl-{int(time.time() * 1000)}"
    threading.Thread(target=_analyze_uploaded_video_audio, args=(path, filename, request_id),
                      name="UploadAudioAnalysis", daemon=True).start()
    return request_id


@app.route("/video/analyze/<filename>")
def video_analyze(filename):
    filename = secure_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "file not found"}), 404
    request_id = _start_upload_analysis(path, filename)
    return jsonify({"ok": True, "status": "processing", "filename": filename,
                    "request_id": request_id})


@app.route("/video/analysis_status")
def video_analysis_status():
    return jsonify(_upload_analysis_snapshot())


def reset_detection_state():
    """Clear the FACE layer (lock + streak) so the next frames are evaluated fresh.

    Without this, switching the video source keeps whatever emotion was
    already locked from the previous source (webcam or a different clip)
    and the UI appears frozen even though new frames are being read.
    A running microphone's "live" voice is untouched here - it keeps running
    regardless of video source. A pinned "upload" voice result IS cleared:
    it describes the file that produced it, so it must not keep outranking
    face for a different file (or the webcam) just because that file never
    got its own voice result (e.g. no speech found).
    """
    clear_emotion_source("face", "detection reset")
    clear_emotion_source("manual", "detection reset")
    if voice_state.get("kind") == "upload":
        clear_emotion_source("voice", "video source changed")


@app.route("/video/use/<filename>")
def video_use(filename):
    global current_video_source
    filename = secure_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "file not found"}), 404
    if not allowed_media_file(filename):
        return jsonify({"ok": False, "error": "unsupported file type"}), 400

    kind = media_kind(filename)
    if kind == "video":
        # Only a file with a picture track may become the camera source.
        print(f"[VIDEO] Switching source -> {path}")
        with video_source_lock:
            current_video_source = path
        reset_detection_state()
    else:
        # Audio-only upload: the camera keeps doing whatever it was doing.
        # Handing a .mp3 to cv2.VideoCapture only produces a dead, blank feed.
        print(f"[VIDEO] {filename} is audio-only - camera source left unchanged")
        # Still a new file though, so a previous file's pinned voice result
        # must not keep outranking face for this one (see reset_detection_state).
        if voice_state.get("kind") == "upload":
            clear_emotion_source("voice", "new upload selected")

    # The (separate, informational) audio analysis runs for BOTH kinds. For a
    # video the FACE layer is already reading its picture track via the camera
    # thread; this reads the same file's soundtrack.
    request_id = _start_upload_analysis(path, filename)
    return jsonify({"ok": True, "source": filename, "kind": kind,
                    "camera_source_changed": kind == "video",
                    "request_id": request_id})


@app.route("/video/use_webcam")
def video_use_webcam():
    global current_video_source
    print("[VIDEO] Switching source -> webcam")
    with video_source_lock:
        current_video_source = 0
    reset_detection_state()
    return jsonify({"ok": True, "source": "webcam"})


@app.route("/video/current")
def video_current():
    with video_source_lock:
        source = current_video_source
    return jsonify({"ok": True, "source": "webcam" if source == 0 else os.path.basename(source)})


@app.route("/status")
def status():
    """Every emotion layer in one response. `emotion`/`song`/`background`/
    `robot_pose_key` always describe the FINAL emotion."""
    snap = emotion_snapshot()
    final = snap["final"]
    emotion = (final["emotion"] or "").lower()
    with video_source_lock:
        video_src = current_video_source

    return jsonify({
        # ---- ordering stamps: every response that carries the fused emotion
        # carries these, so the browser can discard a snapshot that is older
        # than the one already on screen even when it came from another endpoint.
        "state_seq": snap["state_seq"],
        "state_changed_at": snap["state_changed_at"],
        "server_time": time.time(),
        # ---- fused result: what the robot, the music and the UI must follow
        "final": {
            "emotion": emotion.capitalize() if emotion else None,
            "source": final["source"],
            "confidence": round(float(final["confidence"]), 4),
        },
        # ---- raw per-source results, never rewritten by fusion
        "face": {
            "emotion": snap["face"]["emotion"].capitalize() if snap["face"]["emotion"] else None,
            "confidence": round(float(snap["face"]["confidence"]), 4),
            "locked": snap["face"]["locked"],
        },
        "voice": {
            # stable = debounced, the one fusion uses
            "emotion": snap["voice"]["emotion"].capitalize() if snap["voice"]["emotion"] else None,
            "confidence": round(float(snap["voice"]["confidence"]), 4),
            "active": snap["voice"]["active"],
            # "live" (microphone) or "upload" (a completed file analysis) -
            # tells the UI which one currently occupies the voice slot.
            "kind": snap["voice"]["kind"],
            # raw = latest single window, display only
            "raw_emotion": snap["voice_raw"]["emotion"].capitalize() if snap["voice_raw"]["emotion"] else None,
            "raw_confidence": round(float(snap["voice_raw"]["confidence"]), 4),
            "streak": snap["voice_raw"]["streak"],
            "streak_required": snap["voice_raw"]["streak_required"],
            "speech_detected": snap["voice_raw"]["speech_detected"],
        },
        "manual": {
            "emotion": snap["manual"]["emotion"].capitalize() if snap["manual"]["emotion"] else None,
        },
        # ---- backwards-compatible fields (now driven by FINAL, not by the face lock)
        "locked": bool(emotion),
        "emotion": emotion.capitalize() if emotion else None,
        "confidence": round(float(final["confidence"]), 4),
        "source": final["source"],
        # The "default" placeholder is not a detection: it must not start music
        # or repaint the page.
        "song": song_links.get(emotion, "") if (emotion and final["source"] != "default") else "",
        "background": (background_colors.get(emotion, "#ffffff")
                       if final["source"] != "default" else "#ffffff"),
        "robot_connected": robot.connected,
        "robot_busy": robot.busy,
        "robot_pose_key": EMOTION_TO_POSE.get(emotion, "not_mapped") if emotion else "-",
        "last_action": snap["last_action"],
        "detection_active": detection_active,
        "camera_status": _camera_status_snapshot(),
        "video_source": {
            "is_webcam": video_src == 0,
            "filename": "webcam" if video_src == 0 else os.path.basename(video_src),
        },
        "upload_analysis": _upload_analysis_snapshot(),
    })


@app.route("/clear_manual")
def clear_manual():
    """Drop the manual override so FINAL goes back to voice (or face)."""
    clear_emotion_source("manual", "manual override cleared")
    snap = emotion_snapshot()
    return jsonify({"ok": True, "state_seq": snap["state_seq"],
                    "final_emotion": snap["final"]["emotion"],
                    "final_source": snap["final"]["source"]})


@app.route("/reset")
def reset():
    global detection_active
    with lock:
        detection_active = True
    reset_detection_state()
    return jsonify({"status": "reset", "final": emotion_snapshot()["final"]["emotion"]})


@app.route("/stop_detect")
def stop_detect():
    global detection_active
    with lock:
        detection_active = False
    return jsonify({"status": "stopped"})


@app.route("/resume_detect")
def resume_detect():
    global detection_active
    with lock:
        detection_active = True
        history.clear()
    return jsonify({"status": "resumed"})

# ============================================================
# MANUAL EMOTION ROUTES
# Useful because no-TensorFlow OpenCV fallback auto-detects only happy/neutral reliably.
# These routes trigger music/background through /status and robot action.
# ============================================================
@app.route("/set_emotion/<emotion>")
def set_emotion(emotion):
    emotion = emotion.lower().strip()
    known = set(_SONG_CANDIDATES) | set(EMOTION_TO_POSE)
    if emotion not in known:
        return jsonify({"ok": False, "error": f"unknown emotion: {emotion}"}), 400
    lock_emotion_and_trigger(emotion)
    snap = emotion_snapshot()
    final = snap["final"]
    return jsonify({
        "ok": True,
        "state_seq": snap["state_seq"],
        "emotion": emotion,
        "final_emotion": final["emotion"],
        "final_source": final["source"],
        "song": song_links.get(emotion, ""),
        "background": background_colors.get(emotion, "#ffffff"),
        "robot_action": EMOTION_TO_POSE.get(final["emotion"], "not_mapped")
    })

# ============================================================
# ROBOT ROUTES
# ============================================================
@app.route("/robot/scan")
def robot_scan():
    try:
        future = robot_loop.run_async(robot.scan(timeout=8))
        devices = future.result(timeout=12)
        return jsonify({"ok": True, "devices": devices})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/robot/connect", methods=["POST"])
def robot_connect():
    data = request.get_json(silent=True) or {}
    address = data.get("address", "").strip()
    write_index = int(data.get("write_index", 0))

    if not address:
        return jsonify({"ok": False, "error": "address required"})

    try:
        future = robot_loop.run_async(robot.connect(address, write_index))
        ok, msg = future.result(timeout=20)
        return jsonify({"ok": ok, "message": msg})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/robot/disconnect")
def robot_disconnect():
    try:
        future = robot_loop.run_async(robot.disconnect())
        future.result(timeout=8)
        return jsonify({"ok": True, "message": "Robot disconnected"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/robot/status")
def robot_status():
    return jsonify({
        "connected": robot.connected,
        "busy": robot.busy,
        "handle": robot.write_handle,
        "last_device": robot.last_device,
        "last_write_index": robot.last_write_index,
        "last_emotion_executed": robot.last_emotion_executed
    })


@app.route("/robot/move", methods=["POST"])
def robot_move():
    data = request.get_json(silent=True) or {}
    positions = data.get("positions", {})
    move_time = int(data.get("move_time", MOVE_NORMAL))

    try:
        future = robot_loop.run_async(robot.move_servos(pose_to_int_dict(positions), move_time))
        future.result(timeout=10)
        return jsonify({"ok": True, "message": "move command sent"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/robot/home")
def robot_home():
    try:
        send_robot_pose_sync(robot_poses["front_safe"], MOVE_SLOW)
        return jsonify({"ok": True, "message": "home/front safe sent"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/robot/stop")
def robot_stop():
    try:
        future = robot_loop.run_async(robot.stop())
        future.result(timeout=8)
        return jsonify({"ok": True, "message": "stop sent"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/robot/run/<emotion>")
def robot_run_emotion(emotion):
    emotion = emotion.lower().strip()
    if emotion not in EMOTION_TO_POSE:
        return jsonify({"ok": False, "error": f"emotion '{emotion}' not mapped"})
    trigger_robot_for_emotion(emotion)
    return jsonify({"ok": True, "message": f"robot pick started for {emotion}"})


@app.route("/robot/poses")
def robot_get_poses():
    return jsonify({"ok": True, "poses": robot_poses})


@app.route("/robot/save_pose", methods=["POST"])
def robot_save_pose():
    global robot_poses
    data = request.get_json(silent=True) or {}
    pose_key = data.get("pose_key", "").strip()
    pose = data.get("pose", {})

    if not pose_key:
        return jsonify({"ok": False, "error": "pose_key required"})
    if not pose:
        return jsonify({"ok": False, "error": "pose required"})

    clean_pose = {str(k): int(v) for k, v in pose.items()}
    clean_pose["1"] = GRIPPER_OPEN
    robot_poses[pose_key] = clean_pose
    save_robot_poses(robot_poses)

    return jsonify({"ok": True, "message": f"pose saved for {pose_key}", "pose": clean_pose})


@app.route("/process_voice", methods=["POST"])
def process_voice():
    # DEPRECATED: the old speech-to-text + keyword/TextBlob emotion route.
    # Voice emotion now comes from the tone of voice: /voice/start, /voice/stop, /voice/status.
    # Kept only so an old page gets a clear answer. It never sets an emotion or moves the robot.
    return jsonify({
        "ok": False,
        "deprecated": True,
        "error": "/process_voice is deprecated. Use /voice/start, /voice/stop and /voice/status."
    }), 410

# ============================================================
# LIVE VOICE EMOTION (tone of voice, emotion_detector.py)
# Runs in its own worker thread; Flask, camera and BLE threads are not blocked.
# ============================================================
voice_lock = threading.Lock()
voice_emotion = None
voice_confidence = 0.0
voice_probabilities = {"happy": 0.0, "angry": 0.0, "sad": 0.0, "neutral": 0.0}
voice_running = False
voice_speech_detected = False


def on_voice_update(status):
    """Mirror the detector's latest smoothed result (~every 0.5 s) - the LIVE
    microphone layer only.

    This is the *live* reading shown in the voice panel. It does not feed the
    fusion by itself; only the debounced stable emotion does. It does keep
    voice_state["active"] in sync so that stopping the microphone immediately
    hands the final emotion back to the face layer.

    Called continuously (every /voice/status poll) whether or not the mic is
    running, so it must never clear an "upload"-kind voice_state just because
    the mic happens to be off right now - that result is a separate
    observation with its own TTL (see apply_voice_emotion / the "kind" note
    on voice_state above), not something this live-mic mirror owns.
    """
    global voice_emotion, voice_confidence, voice_probabilities, voice_running, voice_speech_detected
    with voice_lock:
        voice_emotion = status["emotion"]
        voice_confidence = status["confidence"]
        voice_probabilities = status["probabilities"]
        voice_running = status["running"]
        voice_speech_detected = status["speech_detected"]

    running = bool(status["running"])
    with lock:
        voice_raw_state.update({
            "emotion": status.get("emotion"),
            "confidence": float(status.get("confidence") or 0.0),
            "streak": round(float(status.get("stability_count") or 0.0), 1),
            "streak_required": int(status.get("stability_required") or 3),
            "speech_detected": bool(status.get("speech_detected")),
        })
        if running:
            voice_state["active"] = True
            if voice_state["emotion"] is None and status.get("stable_emotion"):
                # Re-seed after a page reload: the detector only fires the stable
                # callback when the emotion *changes*, so pick up the current one.
                voice_state.update({
                    "emotion": status["stable_emotion"],
                    "confidence": float(status.get("stable_confidence") or 0.0),
                    "kind": "live", "ts": time.time(),
                })
                print(f"[VOICE] re-seeded stable emotion={voice_state['emotion']}")
            elif (voice_state.get("kind") == "live" and voice_state["emotion"] is not None
                  and float(status.get("stability_count") or 0.0) <= 0.0):
                # The mic is still running, but the live detector's OWN streak has
                # just collapsed to 0 - i.e. it no longer holds ANY confident
                # reading, not even the one currently published. Previously this
                # stale "stable" result stayed authoritative in fusion for the
                # full VOICE_EMOTION_TTL (20s), so the UI could show "Final: ANGRY
                # (from voice)" right next to "streak 0/3" and "speech detected:
                # No" - a live self-contradiction, and it read as "voice emotion
                # isn't updating". Dropping it the moment the streak dies (usually
                # within ~1.5-3s of the tone genuinely changing or stopping, via
                # the per-window decay in emotion_detector._classify_window) hands the
                # final emotion back to face/manual immediately instead of
                # coasting on a reading the live detector itself has abandoned.
                print(f"[VOICE] live streak collapsed - dropping stale "
                      f"{voice_state['emotion']} instead of waiting out the TTL")
                voice_state["emotion"] = None
                voice_state["confidence"] = 0.0
        elif voice_state.get("kind") != "upload":
            # Mic not running and the current voice_state is not an upload
            # result (i.e. it is a live one, or empty): drop it immediately
            # rather than waiting out its TTL, same as before.
            voice_state["active"] = False
            voice_state["emotion"] = None
            voice_state["confidence"] = 0.0
        changed, snapshot = _recompute_final_locked()
    if changed:
        _on_final_emotion_changed(snapshot)


def on_voice_stable_emotion(emotion, confidence, probabilities):
    """Called from the detector worker thread when a new voice emotion becomes stable.

    The debounce lives in emotion_detector.py (N consecutive confident windows),
    so this result is already stabilised. It is pushed into the VOICE layer;
    apply_voice_emotion() re-fuses and, if FINAL changes, drives the robot.
    The face layer is left exactly as it is - its raw prediction stays raw.
    """
    apply_voice_emotion(emotion, confidence)


voice_detector = LiveEmotionDetector(
    input_device=VOICE_INPUT_DEVICE,
    on_stable_emotion=on_voice_stable_emotion,
    on_update=on_voice_update,
)
if VOICE_PRELOAD_MODEL:
    voice_detector.preload_model_async()
atexit.register(voice_detector.stop)


class HideVoicePollingLogs(logging.Filter):
    """Keep the console readable: hide access-log lines for the 500 ms /voice/status polling."""

    def filter(self, record):
        return "/voice/status" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(HideVoicePollingLogs())


@app.route("/voice/start", methods=["GET", "POST"])
def voice_start():
    data = request.get_json(silent=True) or {}
    if "device" in data:
        ok, message = voice_detector.start(input_device=data.get("device"))
    elif request.args.get("device") is not None:
        ok, message = voice_detector.start(input_device=request.args.get("device"))
    else:
        ok, message = voice_detector.start()
    on_voice_update(voice_detector.get_status())
    if not ok:
        return jsonify({"ok": False, "error": message, "message": message})
    return jsonify({"ok": True, "message": message})


@app.route("/voice/stop", methods=["GET", "POST"])
def voice_stop():
    ok, message = voice_detector.stop()
    on_voice_update(voice_detector.get_status())
    return jsonify({"ok": ok, "message": message})


@app.route("/voice/status")
def voice_status():
    status = voice_detector.get_status()
    snap = emotion_snapshot()
    # The robot action shown here is the one taken for the FINAL emotion, so the
    # voice panel and the camera panel can never report different actions.
    status["state_seq"] = snap["state_seq"]
    status["state_changed_at"] = snap["state_changed_at"]
    status["server_time"] = time.time()
    status["last_robot_action"] = snap["last_action"]
    status["final_emotion"] = snap["final"]["emotion"]
    status["final_source"] = snap["final"]["source"]
    status["final_confidence"] = round(float(snap["final"]["confidence"]), 4)
    status["face_emotion"] = snap["face"]["emotion"]
    status["face_locked"] = snap["face"]["locked"]
    status["voice_stable_emotion"] = snap["voice"]["emotion"]
    status["voice_stable_confidence"] = round(float(snap["voice"]["confidence"]), 4)
    status["manual_emotion"] = snap["manual"]["emotion"]
    status["robot_trigger_enabled"] = VOICE_ROBOT_TRIGGER_ENABLED
    status["robot_connected"] = robot.connected
    status["robot_busy"] = robot.busy
    return jsonify(status)


@app.route("/voice/devices")
def voice_devices():
    try:
        return jsonify({
            "ok": True,
            "devices": list_input_devices(),
            "selected": voice_detector.input_device,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "devices": []})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False, threaded=True)