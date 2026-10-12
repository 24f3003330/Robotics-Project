import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from collections import deque
import threading
import time
import json
import asyncio
import atexit
import logging
import traceback
from bleak import BleakScanner, BleakClient
from emotion_detector import LiveEmotionDetector, list_input_devices
from face_emotion import FaceEmotionModel
from voice_gender import LiveGenderTracker
from voice_age.inference import AgeModelUnavailable, LiveAgeTracker
from voice_age.audio import AudioError as VoiceAgeAudioError
from handwriting.inference import HandwritingAnalyzer, HandwritingModelUnavailable
from handwriting.preprocessing import HandwritingInputError
from handwriting import config as handwriting_config
from handwriting import human_collection
from body_vitals.routes import bp as body_vitals_bp

# ============================================================
# NO TENSORFLOW / NO DEEPFACE VERSION
# Python 3.12 friendly version using OpenCV Haar face + smile
# Robot emotion actions are preserved using BLE + taught poses.
#
# Pages:  /        landing page (no recognition)
#         /camera  facial emotion recognition (webcam only)
#         /voice   voice emotion + speaker gender + estimated age group (microphone only)
#         /handwriting  handwriting state from canvas strokes (no camera, no microphone)
#         /body-vitals  WESAD baseline/stress/amusement from ECG, EDA, respiration
#                       (body_vitals/ blueprint; demo/simulated inputs until sensors exist)
# Camera and voice have independent lifecycles: the webcam is opened only
# while a /camera page is viewing it, the microphone and the HuBERT model
# only for /voice. Each is released when its page goes away.
# ============================================================

app = Flask(__name__)
app.register_blueprint(body_vitals_bp)   # /body-vitals - independent of every other modality

# ============================================================
# CAMERA GLOBALS
# ============================================================
latest_frame = None
detection_active = True
lock = threading.Lock()     # guards face_prob_history + every emotion state dict below

# ============================================================
# CAMERA STATUS - what the UI shows instead of a silent black box
# ------------------------------------------------------------
# state is one of: starting | connected | denied | error | stopped
# Written only by the camera thread (CameraService._run); read by /status.
# "denied" is a best-effort guess (OpenCV gives no reliable permission-denied
# signal), reserved for the webcam failing to open.
# ============================================================
camera_status_lock = threading.Lock()
camera_status = {"state": "stopped", "detail": "Camera is off", "source": "webcam"}


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
FACE_EMOTION_TTL = None      # face never expires on its own (it is cleared when lost)
# Face smoothing: every analysed frame's probabilities are averaged over this
# much recent time, so one odd frame cannot flip the emotion but a real change
# of expression shows within about a second.
FACE_SMOOTHING_SECONDS = 1.0
FACE_LOST_SECONDS = 3.0      # no face for this long -> the face result is cleared
FACE_PUBLISH_SECONDS = 0.5   # push the smoothed result into fusion at most this often
face_prob_history = deque()  # (monotonic time, {emotion: p}) - guarded by `lock`
face_model = FaceEmotionModel(log=lambda m: print(f"[FACE MODEL] {m}", flush=True))

# "tracking" = a face is currently being analysed.
face_state = {"emotion": None, "confidence": 0.0, "tracking": False, "probabilities": {},
              "seq": 0, "ts": 0.0}
# voice_state holds the STABLE (debounced) voice emotion from the live
# microphone - the only one that takes part in fusion. voice_raw_state holds
# the latest per-window reading and is for display only, so raw and stable are
# never confused with each other.
voice_state = {"emotion": None, "confidence": 0.0, "active": False, "seq": 0, "ts": 0.0}
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
# Keep False while testing: physical movement is blocked. Set True by operator 
# (via UI / API) to enable real robot action.
HARDWARE_ENABLED = False

# Microphone: None = system default, or an index / name substring listed by
# GET /voice/devices. Can also be chosen in the web page, or set with the
# VOICE_INPUT_DEVICE environment variable.
VOICE_INPUT_DEVICE = os.environ.get("VOICE_INPUT_DEVICE") or None

# The HuBERT model is NOT loaded at start-up: opening /voice starts loading it
# in the background (once per process), so "Start Listening" is quick and the
# camera page never pays for it.

# Page-lifetime watchdogs. A page stops its own pipeline on `pagehide` (via
# navigator.sendBeacon), but a crashed or killed tab never sends that, so each
# pipeline also stops itself once its page has stopped polling for this long.
# The camera page stops polling while hidden, so the webcam is released then
# too; the voice page keeps a slow heartbeat while hidden (browsers throttle
# background timers), hence the longer timeout.
CAMERA_CLIENT_TIMEOUT = 15.0
VOICE_CLIENT_TIMEOUT = 90.0

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
            sid = int(servo_id)
            if sid < 1 or sid > 6:
                raise ValueError(f"Invalid servo ID {sid}. Must be 1-6.")
            pos = int(position)
            if pos < 0 or pos > 1000:
                raise ValueError(f"Invalid position {pos} for servo {sid}. Must be 0-1000.")
            
            params.append(sid)
            params.append(pos & 0xFF)
            params.append((pos >> 8) & 0xFF)

        await self.send_packet(self.SERVO_MOVE_TIME_WRITE, params)

    async def stop(self):
        await self.send_packet(self.SERVO_MOVE_STOP, [])
        self.busy = False


robot_loop = AsyncLoopThread()
robot = XArmBLE()

# ============================================================
# CAMERA SERVICE - the webcam is open only while /camera is being viewed
# ------------------------------------------------------------
# start() is called by the camera page when it loads (and again when it
# becomes visible); stop() when it is hidden or closed. Nothing else opens the
# webcam, so the voice page never touches it. Every /video_frame request
# counts as the page still watching; if they stop arriving for
# CAMERA_CLIENT_TIMEOUT seconds (tab crashed, network gone) the capture thread
# releases the device on its own, so a webcam is never left open with nobody
# looking at it.
# ============================================================
CAMERA_DENIED_MESSAGE = ("Camera access is unavailable. Please grant camera permission and "
                         "try again. (macOS: System Settings > Privacy & Security > Camera - "
                         "allow the app running Python, e.g. Terminal or VS Code.)")


def _open_capture():
    print("[CAMERA] Initializing camera index: webcam(0)")
    _set_camera_status("starting", "Opening webcam ...", "webcam")
    cap = cv2.VideoCapture(0)
    opened = cap.isOpened()
    print(f"[CAMERA] isOpened(): {opened}")
    if not opened:
        print("[CAMERA ERROR] Could not open camera. Check macOS camera permissions "
              "(System Settings > Privacy & Security > Camera)")
        _set_camera_status("denied", CAMERA_DENIED_MESSAGE, "webcam")
        return cap, False

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[VIDEO] FPS = {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"[VIDEO] Resolution = {int(width)}x{int(height)}")

    first_ok, _ = cap.read()
    print(f"[CAMERA] First frame read: {first_ok}")
    if not first_ok:
        print("[CAMERA ERROR] Could not read a frame after opening")
        _set_camera_status("error", "The webcam opened but delivered no picture. It may be in "
                                    "use by another application.", "webcam")
        return cap, False
    print("[CAMERA] Camera started successfully")
    _set_camera_status("connected", "webcam streaming", "webcam")
    return cap, True


class CameraService:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = None
        self._last_client = 0.0

    def touch(self):
        """The camera page is still watching (called on every frame request)."""
        self._last_client = time.monotonic()

    def is_running(self):
        with self._lock:
            return (self._thread is not None and self._thread.is_alive()
                    and not self._stop_event.is_set())

    def start(self):
        """Open the webcam in a background thread. Never blocks the request."""
        with self._lock:
            self.touch()
            previous = self._thread
            if previous is not None and previous.is_alive() and not self._stop_event.is_set():
                return False  # already running
            stop_event = threading.Event()
            thread = threading.Thread(target=self._run, args=(stop_event, previous),
                                      name="CameraCapture", daemon=True)
            self._stop_event = stop_event
            self._thread = thread
            _set_camera_status("starting", "Opening webcam ...", "webcam")
            thread.start()
            return True

    def stop(self, reason="stopped"):
        with self._lock:
            if self._stop_event is None or self._stop_event.is_set():
                return False
            self._stop_event.set()
        print(f"[CAMERA] stop requested ({reason})")
        return True

    def _is_current(self, stop_event):
        with self._lock:
            return self._stop_event is stop_event

    def _run(self, stop_event, previous):
        global latest_frame
        # A previous capture that is still shutting down must release the
        # device before this one opens it.
        if previous is not None:
            previous.join(timeout=3.0)
        # The facial emotion model is loaded here - only when the camera page
        # starts the camera, once per process - never by the voice page.
        _set_camera_status("starting", "Loading facial emotion model ...", "webcam")
        face_model.ensure_loaded()
        cap, ok = _open_capture()
        frame_counter = 0
        consecutive_failures = 0
        reason = "stopped by the page"
        try:
            if not ok:
                reason = None  # keep the denied/error status for the UI
                return
            while not stop_event.is_set():
                if time.monotonic() - self._last_client > CAMERA_CLIENT_TIMEOUT:
                    reason = f"no viewer for {CAMERA_CLIENT_TIMEOUT:.0f}s"
                    print(f"[CAMERA] {reason} - releasing the webcam")
                    break

                success, frame = cap.read()
                latest_frame = frame if success and frame is not None else None

                cam_snap = _camera_status_snapshot()
                if success and frame is not None:
                    consecutive_failures = 0
                    if cam_snap["state"] != "connected":
                        _set_camera_status("connected", "webcam streaming", "webcam")
                else:
                    consecutive_failures += 1
                    # Many failed reads in a row (not just one dropped frame) means
                    # the device really did go away mid-session - surface that
                    # instead of silently freezing on the last good frame forever.
                    if consecutive_failures == 100 and cam_snap["state"] == "connected":
                        print("[CAMERA ERROR] Lost the video stream (no frames for ~1s)")
                        _set_camera_status("error", "Lost the video stream. Check that the "
                                                    "webcam is still connected.", "webcam")

                frame_counter += 1
                if frame_counter % 60 == 1:
                    print(f"[DETECTION] Processing frame #{frame_counter} "
                          f"(ok={success and frame is not None})")

                time.sleep(0.01)
        except Exception as exc:
            reason = None
            print(f"[CAMERA ERROR] capture loop failed: {exc}")
            traceback.print_exc()
            _set_camera_status("error", f"Camera capture failed: {exc}", "webcam")
        finally:
            cap.release()
            stop_event.set()
            if self._is_current(stop_event):
                latest_frame = None
                if reason:
                    _set_camera_status("stopped", "Camera is off", "webcam")
                # The face result describes a camera that is no longer running.
                clear_emotion_source("face", "camera stopped")
            print("[CAMERA] webcam released")


camera_service = CameraService()

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

    if not HARDWARE_ENABLED:
        print(f"[DRY-RUN] Would execute robot pick for emotion: {emotion}")
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
        if source == "voice" and not state["active"]:
            continue  # microphone stopped -> voice drops out of the fusion
        ttl = SOURCE_TTL[source]
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

    # Face, voice, and manual trigger the physical sequence only if HARDWARE_ENABLED.
    if not HARDWARE_ENABLED:
        action = f"simulated {emotion}"
        print(f"[DRY RUN / ROBOT SIMULATION] Would trigger {emotion} from {source}")
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
        previous_emotion = state["emotion"]
        state.update({
            "emotion": emotion,
            "confidence": float(confidence),
            "seq": _emotion_seq,
            "ts": observed_at,
        })
        if extra:
            state.update(extra)
        # The face layer is refreshed twice a second; only log it when it changes.
        if source != "face" or emotion != previous_emotion:
            tag = {"face": "FACE", "voice": "VOICE STABLE->APP", "manual": "MANUAL"}[source]
            print(f"[{tag}] emotion={emotion.upper()} confidence={float(confidence) * 100:.0f}",
                  flush=True)
        changed, snapshot = _recompute_final_locked()

    if changed:
        _on_final_emotion_changed(snapshot)
    return snapshot


def apply_face_emotion(emotion, confidence, probabilities=None, observed_at=None):
    """Smoothed webcam result. Stays raw: fusion may ignore it, but it is never rewritten."""
    return _apply_emotion("face", emotion, confidence,
                          extra={"tracking": True, "probabilities": dict(probabilities or {})},
                          observed_at=observed_at)


def apply_voice_emotion(emotion, confidence, observed_at=None):
    """Stable tone-of-voice result from the live microphone (emotion_detector.py)."""
    return _apply_emotion("voice", emotion, confidence,
                          extra={"active": True}, observed_at=observed_at)


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
    """Drop one source's result (e.g. face when the camera stops) and re-fuse."""
    with lock:
        SOURCE_STATES[source].update({"emotion": None, "confidence": 0.0, "seq": 0, "ts": 0.0})
        if source == "face":
            face_state["tracking"] = False
            face_state["probabilities"] = {}
            face_prob_history.clear()
        print(f"[{source.upper()}] cleared{(' - ' + reason) if reason else ''}")
        changed, snapshot = _recompute_final_locked()
    if changed:
        _on_final_emotion_changed(snapshot)
    return snapshot


# ============================================================
# VIDEO STREAM
# ============================================================
# Counter for the "log every 30th frame" cadence in _render_detection_frame.
_frame_log_counter = 0
_frame_log_lock = threading.Lock()


def _next_frame_log_index():
    global _frame_log_counter
    with _frame_log_lock:
        _frame_log_counter += 1
        return _frame_log_counter


def _placeholder_frame(lines, color=(200, 200, 200)):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, 220 + 40 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    ret, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()


def _render_and_encode_frame(quality=80):
    """Read latest_frame, run detection/overlay on a COPY of it, return JPEG bytes."""
    frame = latest_frame

    if frame is None:
        state = _camera_status_snapshot()["state"]
        if state in ("denied", "error"):
            return _placeholder_frame(["Camera unavailable",
                                       "Check camera permission and retry"], (80, 80, 255))
        if state == "starting":
            return _placeholder_frame(["Starting camera..."])
        return _placeholder_frame(["Camera is off"])

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


_face_last_seen = 0.0
_face_last_publish = 0.0


def _update_face_emotion(result, now):
    """Fold one frame's analysis into the smoothed FACE layer.

    Probabilities are averaged over the last FACE_SMOOTHING_SECONDS, the
    argmax becomes the face emotion, and the result is pushed into fusion at
    most every FACE_PUBLISH_SECONDS (immediately when the emotion changes).
    A face that has been gone for FACE_LOST_SECONDS is cleared rather than
    left showing a stale emotion.
    """
    global _face_last_seen, _face_last_publish
    if result is None:
        with lock:
            lost = face_state["tracking"] and now - _face_last_seen > FACE_LOST_SECONDS
        if lost:
            clear_emotion_source("face", f"no face for {FACE_LOST_SECONDS:.0f}s")
        return None

    with lock:
        _face_last_seen = now
        face_prob_history.append((now, result["probabilities"]))
        while face_prob_history and now - face_prob_history[0][0] > FACE_SMOOTHING_SECONDS:
            face_prob_history.popleft()
        labels = result["probabilities"].keys()
        n = len(face_prob_history)
        smoothed = {e: sum(p.get(e, 0.0) for _, p in face_prob_history) / n for e in labels}
        emotion = max(smoothed, key=smoothed.get)
        due = (emotion != face_state["emotion"]
               or now - _face_last_publish >= FACE_PUBLISH_SECONDS)
        if due:
            _face_last_publish = now
    if due:
        apply_face_emotion(emotion, smoothed[emotion],
                           {e: round(p, 4) for e, p in smoothed.items()})
    return emotion, smoothed[emotion]


def _render_detection_frame(frame, face_log_counter):
    """Detect the face, classify its emotion and draw the overlay, in place on `frame`.

    Split out of _render_and_encode_frame() so the whole thing can be wrapped
    in one try/except there without a deep nested indent change.
    """
    if not detection_active:
        cv2.putText(frame, "Detection Paused", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        now = time.monotonic()
        result = face_model.analyze(frame)
        smoothed = _update_face_emotion(result, now)
        if face_log_counter % 30 == 1:
            if result is None:
                print("[FACE] no face in frame", flush=True)
            else:
                top = sorted(result["probabilities"].items(), key=lambda kv: -kv[1])[:3]
                print(f"[FACE] frame={result['emotion'].upper()} "
                      f"({' '.join(f'{e}={p:.2f}' for e, p in top)}) "
                      f"-> smoothed={smoothed[0].upper()} {smoothed[1] * 100:.0f}%", flush=True)

        if result is None:
            cv2.putText(frame, "No face detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            x, y, w, h = result["box"]
            thickness = max(2, frame.shape[1] // 400)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            label = f"{smoothed[0].upper()} {smoothed[1] * 100:.0f}%"
            scale = max(0.9, frame.shape[1] / 1000.0)
            cv2.putText(frame, label, (x, max(30, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness)

    # Always show the fused result on the frame so the overlay can never
    # disagree with the rest of the UI.
    final_label = final_state["emotion"]
    final_source = final_state["source"]
    overlay = f"Final: {final_label.upper()} ({final_source})" if final_label else "Final: -"
    cv2.putText(frame, overlay, (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.8, frame.shape[1] / 1400.0), (0, 215, 255), 2)

# ============================================================
# PAGES
# ============================================================
@app.route("/")
def landing():
    """Landing page. Starts nothing: no camera, no microphone, no model."""
    return render_template("landing.html", active_page="home")


@app.route("/camera")
def camera():
    global detection_active
    with lock:
        detection_active = True
    # Fresh camera/manual state for a fresh page. The webcam itself is opened
    # by the page's JS (POST /camera/start), never by a page render.
    clear_emotion_source("face", "camera page loaded")
    clear_emotion_source("manual", "camera page loaded")
    return render_template("camera.html", active_page="camera")


@app.route("/voice")
def voice():
    clear_emotion_source("manual", "voice page loaded")
    # Start loading HuBERT in the background (once per process) so it is ready
    # by the time the user presses Start Listening.
    voice_detector.preload_model_async()
    voice_gender.preload_async()
    voice_age.preload_async()
    on_voice_update(voice_detector.get_status())
    from voice_age import config as voice_age_config
    return render_template("voice.html", active_page="voice",
                           age_window_seconds=voice_age_config.AGE_INFERENCE_WINDOW_SECONDS,
                           age_disclaimer=voice_age_config.DISCLAIMER)


# ============================================================
# CAMERA ROUTES
# ============================================================
@app.route("/camera/start", methods=["POST"])
def camera_start():
    started = camera_service.start()
    return jsonify({"ok": True, "started": started, "camera_status": _camera_status_snapshot()})


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    """Also the target of the page's navigator.sendBeacon() on pagehide."""
    stopped = camera_service.stop("camera page closed or hidden")
    return jsonify({"ok": True, "stopped": stopped})


@app.route("/video_frame")
def video_frame():
    """One current JPEG frame with the detection overlay already drawn, fetched
    by the camera page on a JS interval.
    ------------------------------------------------------------------
    Why polling rather than an MJPEG <img>: Safari/WebKit does not reliably
    render multipart/x-mixed-replace in an <img> (Chrome and Firefox do). A
    single-JPEG-per-request endpoint works identically in every browser.
    ------------------------------------------------------------------
    Every request also tells CameraService the page is still watching.
    """
    camera_service.touch()
    frame_bytes = _render_and_encode_frame()
    resp = Response(frame_bytes, mimetype="image/jpeg")
    # Every poll must fetch a fresh frame - a cached one would freeze the preview.
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


def reset_detection_state():
    """Clear the FACE layer (lock + streak) and any manual override so the next
    frames are evaluated fresh. A running microphone's voice layer is untouched."""
    clear_emotion_source("face", "detection reset")
    clear_emotion_source("manual", "detection reset")


def _final_output_fields(snap):
    """What the FINAL emotion drives: music, background and robot pose."""
    final = snap["final"]
    emotion = (final["emotion"] or "").lower()
    detected = bool(emotion) and final["source"] != "default"
    return {
        # The "default" placeholder is not a detection: it must not start music
        # or repaint the page.
        "song": song_links.get(emotion, "") if detected else "",
        "background": background_colors.get(emotion, "#ffffff") if detected else "#ffffff",
        "robot_pose_key": EMOTION_TO_POSE.get(emotion, "not_mapped") if emotion else "-",
    }


@app.route("/status")
def status():
    """Camera page status: every emotion layer in one response.
    `emotion`/`song`/`background`/`robot_pose_key` always describe the FINAL emotion."""
    snap = emotion_snapshot()
    final = snap["final"]
    emotion = (final["emotion"] or "").lower()

    return jsonify({
        # ---- ordering stamps: every response that carries the fused emotion
        # carries these, so the browser can discard a snapshot that is older
        # than the one already on screen.
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
            "tracking": snap["face"]["tracking"],
            "probabilities": snap["face"]["probabilities"],
            "model": face_model.kind,
            "model_warning": face_model.warning,
        },
        "manual": {
            "emotion": snap["manual"]["emotion"].capitalize() if snap["manual"]["emotion"] else None,
        },
        # ---- backwards-compatible fields (driven by FINAL)
        "locked": bool(emotion),
        "emotion": emotion.capitalize() if emotion else None,
        "confidence": round(float(final["confidence"]), 4),
        "source": final["source"],
        **_final_output_fields(snap),
        "robot_connected": robot.connected,
        "robot_busy": robot.busy,
        "last_action": snap["last_action"],
        "detection_active": detection_active,
        "camera_running": camera_service.is_running(),
        "camera_status": _camera_status_snapshot(),
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
        face_prob_history.clear()
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
        "last_emotion_executed": robot.last_emotion_executed,
        "hardware_enabled": HARDWARE_ENABLED
    })


@app.route("/robot/move", methods=["POST"])
def robot_move():
    data = request.get_json(silent=True) or {}
    positions = data.get("positions", {})
    move_time = int(data.get("move_time", MOVE_NORMAL))

    try:
        if not HARDWARE_ENABLED:
            return jsonify({"ok": True, "message": "[DRY RUN] move command sent (simulated)"})

        future = robot_loop.run_async(robot.move_servos(pose_to_int_dict(positions), move_time))
        future.result(timeout=10)
        return jsonify({"ok": True, "message": "move command sent; completion unverified"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/robot/home")
def robot_home():
    try:
        if not HARDWARE_ENABLED:
            return jsonify({"ok": True, "message": "[DRY RUN] home/front safe sent (simulated)"})
        
        send_robot_pose_sync(robot_poses["front_safe"], MOVE_SLOW)
        return jsonify({"ok": True, "message": "home/front safe sent; completion unverified"})
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
    
    if not HARDWARE_ENABLED:
        return jsonify({"ok": True, "message": f"[DRY RUN] robot pick started for {emotion}"})
    
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


@app.route("/robot/hardware_enable", methods=["POST"])
def robot_hardware_enable():
    global HARDWARE_ENABLED
    data = request.get_json(silent=True) or {}
    enable = data.get("enable", False)
    HARDWARE_ENABLED = bool(enable)
    print(f"[ROBOT] Hardware Enabled set to {HARDWARE_ENABLED}")
    return jsonify({"ok": True, "hardware_enabled": HARDWARE_ENABLED})

# ============================================================
# LIVE VOICE EMOTION (tone of voice, emotion_detector.py)
# ------------------------------------------------------------
#   sounddevice mic callback -> bounded queue -> VoiceEmotionWorker thread
#     -> Silero VAD -> speech accumulator (2-4 s windows) -> HuBERT
#     -> smoothing / debounce -> on_update / on_stable_emotion (below)
#   /voice page polls /voice/status every 500 ms and renders every prediction.
# Runs in its own worker thread; Flask, camera and BLE threads are not blocked.
#
# Speaker gender (voice_gender.py) shares that microphone stream and VAD: the
# worker hands it every VAD chunk (on_vad_chunk), and it classifies 1.5-3 s
# speech segments on its own VoiceGenderWorker thread. It never feeds fusion,
# music or the robot, and a gender failure never affects voice emotion.
#
# Estimated AGE GROUP (voice_age/) works the same way: a separate frozen encoder
# + classifier trained on Common Voice, fed the same VAD chunks, classifying 3 s
# speech windows on its own VoiceAgeWorker thread. It is display-only and a
# failure (or a model that was never trained) never affects emotion or gender.
# ============================================================
MICROPHONE_DENIED_MESSAGE = ("Microphone access is unavailable. Please grant microphone "
                             "permission and try again.")


def on_voice_update(status):
    """Mirror the detector's latest result (~every 0.5 s) into the VOICE layer.

    The per-window reading is display-only (voice_raw_state); only the
    debounced stable emotion takes part in fusion. This keeps
    voice_state["active"] in sync so that stopping the microphone immediately
    hands the final emotion back to the other sources.
    """
    running = bool(status["running"])
    if not running:
        voice_gender.stop_session()     # the microphone is gone, whoever stopped it
        voice_age.stop_session()
    with lock:
        voice_raw_state.update({
            "emotion": status.get("emotion"),
            "confidence": float(status.get("confidence") or 0.0),
            "streak": round(float(status.get("stability_count") or 0.0), 1),
            "streak_required": status.get("stability_required") or 3,
            "speech_detected": bool(status.get("speech_detected")),
        })
        stability = float(status.get("stability_count") or 0.0)
        required = float(status.get("stability_required") or 0.0)
        if running:
            voice_state["active"] = True
            if (voice_state["emotion"] is None and status.get("stable_emotion")
                    and required > 0 and stability >= required):
                # Re-seed after a page reload: the detector only fires the stable
                # callback when the emotion *changes*, so pick up the current one.
                # Only while the detector still genuinely holds it (streak at
                # threshold): re-seeding from a stale stable_emotion undid the
                # "streak collapsed" drop below on the very next update, so the
                # final emotion flip-flopped every 0.5 s.
                voice_state.update({
                    "emotion": status["stable_emotion"],
                    "confidence": float(status.get("stable_confidence") or 0.0),
                    "ts": time.time(),
                })
                print(f"[VOICE] re-seeded stable emotion={voice_state['emotion']}")
            elif voice_state["emotion"] is not None and stability <= 0.0:
                # The mic is still running, but the live detector's OWN streak has
                # collapsed to 0 - it no longer holds ANY confident reading. Drop
                # the stale result now instead of letting it stay authoritative
                # for the full VOICE_EMOTION_TTL.
                print(f"[VOICE] live streak collapsed - dropping stale "
                      f"{voice_state['emotion']} instead of waiting out the TTL")
                voice_state["emotion"] = None
                voice_state["confidence"] = 0.0
        else:
            # Mic not running: drop the voice result immediately.
            voice_state["active"] = False
            voice_state["emotion"] = None
            voice_state["confidence"] = 0.0
        changed, snapshot = _recompute_final_locked()
    if changed:
        _on_final_emotion_changed(snapshot)


def on_voice_stable_emotion(emotion, confidence, probabilities):
    """Called from the detector worker thread when a new voice emotion becomes stable.

    The debounce lives in emotion_detector.py, so this result is already
    stabilised. apply_voice_emotion() re-fuses and, if FINAL changes, drives
    the robot.
    """
    apply_voice_emotion(emotion, confidence)


# The gender model loads when /voice opens (never for /camera), and only after
# the emotion model: both import transformers, whose imports are not thread-safe.
voice_gender = LiveGenderTracker(load_after=lambda: voice_detector._ensure_model())
# Age loads after both other models (same transformers import-safety reason).
voice_age = LiveAgeTracker(load_after=lambda: (voice_detector._ensure_model(), voice_gender._ensure_model()))


def on_voice_vad_chunk(chunk, is_speech):
    """The single microphone stream's VAD chunks fan out to the gender and age trackers.
    Each feed() is cheap and never raises, so neither can disturb the other or emotion."""
    voice_gender.feed(chunk, is_speech)
    voice_age.feed(chunk, is_speech)


voice_detector = LiveEmotionDetector(
    input_device=VOICE_INPUT_DEVICE,
    on_stable_emotion=on_voice_stable_emotion,
    on_update=on_voice_update,
    on_vad_chunk=on_voice_vad_chunk,
)
atexit.register(voice_detector.stop)


# ------------------------------------------------------------
# Voice page watchdog: stops the microphone when the /voice page that started
# it stops polling (closed without a pagehide beacon, crashed, lost network).
# At most one watchdog thread exists; it exits as soon as the detector stops.
# ------------------------------------------------------------
_voice_client_seen = 0.0
_voice_watchdog_lock = threading.Lock()
_voice_watchdog_thread = None


def _touch_voice_client():
    global _voice_client_seen
    _voice_client_seen = time.monotonic()


def _voice_watchdog():
    while voice_detector.is_running():
        time.sleep(2.0)
        idle = time.monotonic() - _voice_client_seen
        if idle > VOICE_CLIENT_TIMEOUT and voice_detector.is_running():
            print(f"[VOICE] no /voice page has polled for {idle:.0f}s - stopping the microphone")
            voice_detector.stop()
            on_voice_update(voice_detector.get_status())
            break


def _ensure_voice_watchdog():
    global _voice_watchdog_thread
    with _voice_watchdog_lock:
        if _voice_watchdog_thread is None or not _voice_watchdog_thread.is_alive():
            _voice_watchdog_thread = threading.Thread(target=_voice_watchdog,
                                                      name="VoicePageWatchdog", daemon=True)
            _voice_watchdog_thread.start()


class HidePollingLogs(logging.Filter):
    """Keep the console readable: hide access-log lines for the pages' polling."""

    POLLED = ("/voice/status", "/video_frame", "/status")

    def filter(self, record):
        message = record.getMessage()
        return not any(f"GET {path} " in message for path in self.POLLED)


logging.getLogger("werkzeug").addFilter(HidePollingLogs())


@app.route("/voice/start", methods=["POST"])
def voice_start():
    _touch_voice_client()
    data = request.get_json(silent=True) or {}
    if "device" in data:
        ok, message = voice_detector.start(input_device=data.get("device"))
    else:
        ok, message = voice_detector.start()
    on_voice_update(voice_detector.get_status())
    if not ok:
        return jsonify({"ok": False, "error": message, "message": message,
                        "user_message": MICROPHONE_DENIED_MESSAGE})
    voice_gender.start_session()
    voice_age.start_session()
    _ensure_voice_watchdog()
    return jsonify({"ok": True, "message": message})


@app.route("/voice/stop", methods=["POST"])
def voice_stop():
    """Also the target of the page's navigator.sendBeacon() on pagehide."""
    ok, message = voice_detector.stop()
    on_voice_update(voice_detector.get_status())
    return jsonify({"ok": ok, "message": message})


@app.route("/voice/status")
def voice_status():
    _touch_voice_client()
    status = voice_detector.get_status()
    snap = emotion_snapshot()
    # The robot action shown here is the one taken for the FINAL emotion.
    status["state_seq"] = snap["state_seq"]
    status["state_changed_at"] = snap["state_changed_at"]
    status["server_time"] = time.time()
    status["last_robot_action"] = snap["last_action"]
    status["final_emotion"] = snap["final"]["emotion"]
    status["final_source"] = snap["final"]["source"]
    status["final_confidence"] = round(float(snap["final"]["confidence"]), 4)
    status["voice_stable_emotion"] = snap["voice"]["emotion"]
    status["voice_stable_confidence"] = round(float(snap["voice"]["confidence"]), 4)
    status["manual_emotion"] = snap["manual"]["emotion"]
    status["hardware_enabled"] = HARDWARE_ENABLED
    status["robot_connected"] = robot.connected
    status["robot_busy"] = robot.busy
    status.update(_final_output_fields(snap))
    if not status["running"]:
        voice_gender.stop_session()     # no-op unless a stop raced a start
        voice_age.stop_session()
    status["gender"] = voice_gender.get_status()
    try:
        status["age"] = voice_age.get_status()
    except Exception as exc:            # age must never break the emotion status the page relies on
        status["age"] = {"state": "unavailable", "message": f"age status failed: {exc}"}
    return jsonify(status)


@app.route("/voice/age/status")
def voice_age_status():
    """Is the voice age-group model trained / loaded? (never loads it, never fails)."""
    return jsonify(voice_age.estimator.status())


VOICE_AGE_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@app.route("/voice/age", methods=["POST"])
def voice_age_estimate():
    """Estimated age GROUP for one recorded clip (multipart field "audio": WAV / FLAC / OGG / MP3).
    The live /voice page does not use this: it gets "age" from /voice/status, computed from the
    same microphone stream as emotion."""
    if (request.content_length or 0) > VOICE_AGE_MAX_UPLOAD_BYTES:
        return jsonify({"success": False, "kind": "input", "error": "Audio file too large (max 20 MB)."}), 413
    upload = request.files.get("audio")
    if upload is None:
        return jsonify({"success": False, "kind": "input", "error": "No audio uploaded (form field 'audio')."}), 400
    data = upload.read(VOICE_AGE_MAX_UPLOAD_BYTES + 1)
    try:
        from voice_age.audio import decode
        samples, rate = decode(data)
        result = voice_age.estimator.predict(samples, rate)
    except VoiceAgeAudioError as exc:
        return jsonify({"success": False, "kind": "insufficient_audio", "reason": exc.reason,
                        "error": f"Not enough usable speech: {exc}"}), 422
    except AgeModelUnavailable as exc:
        return jsonify({"success": False, "kind": "model_unavailable", "error": str(exc),
                        "status": voice_age.estimator.status()}), 503
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "kind": "error", "error": f"Age estimation failed: {exc}"}), 500
    from voice_age import config as voice_age_config
    result.update(success=True, label=voice_age_config.UI_LABEL, disclaimer=voice_age_config.DISCLAIMER)
    return jsonify(result)


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


# ============================================================
# HANDWRITING (handwriting/ package)
# ------------------------------------------------------------
#   canvas strokes (POST /handwriting/analyze) -> validate -> resample ->
#   per-point features -> Transformer + attention pooling -> one head per
#   EMOTHAW target: Anxiety / Depression / Stress, each normal vs elevated
# Independent of camera and voice: no webcam, no microphone, no fusion, no
# music or robot. The checkpoint (models/handwriting_transformer.pt) is loaded
# once, in the background when /handwriting opens; without it the page says the
# model is not installed instead of predicting anything.
#
# A SEPARATE 4-class experiment (Happy / Neutral / Sad / Angry) has its own
# checkpoint (models/handwriting_4class.pt) and routes (/handwriting/4class/*).
# It is currently trained on graphology-based SEED data - a prototype, not a
# validated emotion model - and its responses say so ("prototype": true).
# ============================================================
handwriting_analyzer = HandwritingAnalyzer()
handwriting_4class_analyzer = HandwritingAnalyzer(handwriting_config.FOUR_CLASS_CHECKPOINT, kind="4class")
HANDWRITING_MAX_REQUEST_BYTES = 20 * 1024 * 1024


@app.route("/handwriting")
def handwriting_page():
    handwriting_analyzer.preload_async()
    handwriting_4class_analyzer.preload_async()
    # open on a tab that can actually predict: the EMOTHAW model needs a dataset that is
    # not shipped, so without it the page starts on the 4-class prototype
    emothaw_ready = os.path.isfile(handwriting_config.DEFAULT_CHECKPOINT)
    four_ready = os.path.isfile(handwriting_config.FOUR_CLASS_CHECKPOINT)
    initial_mode = "4class" if four_ready and not emothaw_ready else "emothaw"
    return render_template("handwriting.html", active_page="handwriting", initial_mode=initial_mode,
                           prototype_notice=handwriting_config.PROTOTYPE_NOTICE,
                           four_class_labels=handwriting_config.FOUR_CLASS_LABELS)


@app.route("/handwriting/status")
def handwriting_status():
    return jsonify(handwriting_analyzer.status())


@app.route("/handwriting/analyze", methods=["POST"])
def handwriting_analyze():
    return _handwriting_analyze_with(handwriting_analyzer)


@app.route("/handwriting/4class/status")
def handwriting_4class_status():
    return jsonify(handwriting_4class_analyzer.status())


@app.route("/handwriting/4class/analyze", methods=["POST"])
def handwriting_4class_analyze():
    return _handwriting_analyze_with(handwriting_4class_analyzer)


@app.route("/handwriting/4class/analyze-image", methods=["POST"])
def handwriting_4class_analyze_image():
    """4-class prototype: multipart upload (field "image": PNG / JPG / WEBP) -> image_trajectory
    reconstruction -> the same preprocessing and model as canvas input. Same response
    structure as /handwriting/4class/analyze, plus input.source_type = "uploaded_image"."""
    from handwriting.inference import MAX_UPLOAD_BYTES
    analyzer = handwriting_4class_analyzer
    if (request.content_length or 0) > MAX_UPLOAD_BYTES + 64 * 1024:
        return jsonify({"ok": False, "kind": "input",
                        "error": f"The image is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."}), 413
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"ok": False, "kind": "input", "error": "No image was uploaded (form field 'image')."}), 400
    data = upload.read(MAX_UPLOAD_BYTES + 1)
    try:
        result = analyzer.analyze_image(data)
    except HandwritingInputError as exc:
        return jsonify({"ok": False, "kind": "input", "error": str(exc)}), 400
    except HandwritingModelUnavailable as exc:
        return jsonify({"ok": False, "kind": "model_unavailable", "error": str(exc),
                        "status": analyzer.status()}), 503
    except Exception as exc:
        print(f"[HANDWRITING ERROR] image analysis failed: {exc}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "kind": "error", "error": f"Handwriting analysis failed: {exc}"}), 500
    print(f"[HANDWRITING 4-class image] {upload.filename!r} {result['input']['image_size']} -> "
          f"{result['emotion']} {result['confidence']:.0%} ({result['inference_ms']} ms)", flush=True)
    return jsonify(result)


@app.route("/handwriting/4class/collect", methods=["POST"])
def handwriting_4class_collect():
    """Collect tab: save one participant sample (anonymous writer id + chosen label)
    as raw canvas JSON under data/handwriting/seed_4class/human/<label>/."""
    if (request.content_length or 0) > HANDWRITING_MAX_REQUEST_BYTES:
        return jsonify({"ok": False, "error": "Too much stroke data in one request."}), 413
    payload = request.get_json(silent=True)
    try:
        saved = human_collection.save_sample(payload)
    except HandwritingInputError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    print(f"[HANDWRITING COLLECT] {saved['writer_id']} {saved['label']} #{saved['number']} -> {saved['file']}",
          flush=True)
    return jsonify(dict(saved, ok=True))


@app.route("/handwriting/4class/collect/status")
def handwriting_4class_collect_status():
    writer_id = request.args.get("writer_id") or None
    try:
        writer_id = human_collection.check_writer_id(writer_id) if writer_id else None
    except HandwritingInputError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify(dict(human_collection.counts(writer_id=writer_id), ok=True,
                        next_writer_id=human_collection.next_writer_id()))


def _handwriting_analyze_with(analyzer):
    if (request.content_length or 0) > HANDWRITING_MAX_REQUEST_BYTES:
        return jsonify({"ok": False, "kind": "input", "error": "Too much stroke data in one request."}), 413
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"ok": False, "kind": "input",
                        "error": "Expected a JSON body with a 'points' list."}), 400
    try:
        result = analyzer.analyze(payload)
    except HandwritingInputError as exc:
        return jsonify({"ok": False, "kind": "input", "error": str(exc)}), 400
    except HandwritingModelUnavailable as exc:
        return jsonify({"ok": False, "kind": "model_unavailable", "error": str(exc),
                        "input": getattr(exc, "input_summary", None),
                        "status": analyzer.status()}), 503
    except Exception as exc:
        print(f"[HANDWRITING ERROR] analysis failed: {exc}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "kind": "error", "error": f"Handwriting analysis failed: {exc}"}), 500
    heads = ", ".join(f"{h['name']}={h['label']} {h['confidence']:.0%}" for h in result["heads"])
    print(f"[HANDWRITING{' 4-class' if analyzer.kind == '4class' else ''}] {result['input']['points_received']} points, {result['input']['strokes']} "
          f"strokes -> {heads} ({result['inference_ms']} ms)", flush=True)
    return jsonify(result)


atexit.register(camera_service.stop, "server shutting down")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False, threaded=True)
