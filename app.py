import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from collections import deque
import threading
import time
import json
import asyncio
from bleak import BleakScanner, BleakClient

# ============================================================
# NO TENSORFLOW / NO DEEPFACE VERSION
# Python 3.12 friendly version using OpenCV Haar face + smile
# Robot emotion actions are preserved using BLE + taught poses.
# ============================================================

app = Flask(__name__)

# ============================================================
# CAMERA + SIMPLE EMOTION GLOBALS
# ============================================================
latest_frame = None
camera_running = True
current_emotion = None
emotion_locked = False
detection_active = True
history = deque(maxlen=5)
lock = threading.Lock()

# OpenCV built-in Haar cascades. No TensorFlow required.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# Existing music mapping retained.
song_links = {
    "happy": "/static/music/happy.m4a",
    "sad": "/static/music/sad.m4a",
    "angry": "/static/music/angry.m4a",
    "fear": "/static/music/fear.m4a",
    "surprise": "/static/music/surprise.m4a",
    "disgust": "/static/music/disgust.m4a",
    "neutral": "/static/music/neutral.m4a",
    "contempt": "/static/music/contempt.m4a",
}

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
def update_camera():
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera did not open")
    else:
        print("Camera opened successfully")

    while camera_running:
        success, frame = cap.read()
        latest_frame = frame if success and frame is not None else None
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
    global current_emotion, emotion_locked
    with lock:
        current_emotion = emotion
        emotion_locked = True
    trigger_robot_for_emotion(emotion)

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
# VIDEO STREAM
# ============================================================
def generate_frames():
    global current_emotion, emotion_locked, detection_active, latest_frame

    while True:
        frame = latest_frame

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Error: Cannot read webcam", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Check camera privacy settings", (20, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.1)
            continue

        frame = frame.copy()

        if not detection_active:
            cv2.putText(frame, "Detection Paused", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        elif not emotion_locked:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_gray = gray[y:y+h, x:x+w]

                    detected = simple_opencv_emotion(face_gray)
                    with lock:
                        history.append(detected)
                        if len(history) == history.maxlen and len(set(history)) == 1:
                            current_emotion = detected
                            emotion_locked = True
                            trigger_robot_for_emotion(detected)

                    label = current_emotion if current_emotion else detected
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        else:
            label = current_emotion if current_emotion else "Locked"
            cv2.putText(frame, f"Locked: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# ============================================================
# EXISTING FLASK ROUTES
# ============================================================
@app.route("/")
def index():
    global current_emotion, emotion_locked, detection_active
    with lock:
        current_emotion = None
        emotion_locked = False
        detection_active = True
        history.clear()
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with lock:
        if emotion_locked and current_emotion:
            emotion = current_emotion.lower()
            return jsonify({
                "locked": True,
                "emotion": emotion.capitalize(),
                "song": song_links.get(emotion, ""),
                "background": background_colors.get(emotion, "#ffffff"),
                "robot_connected": robot.connected,
                "robot_busy": robot.busy,
                "robot_pose_key": EMOTION_TO_POSE.get(emotion, "not_mapped")
            })
        return jsonify({
            "locked": False,
            "robot_connected": robot.connected,
            "robot_busy": robot.busy
        })


@app.route("/reset")
def reset():
    global current_emotion, emotion_locked, detection_active
    with lock:
        current_emotion = None
        emotion_locked = False
        detection_active = True
        history.clear()
    return jsonify({"status": "reset"})


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
    if emotion not in song_links:
        return jsonify({"ok": False, "error": f"unknown emotion: {emotion}"})
    lock_emotion_and_trigger(emotion)
    return jsonify({
        "ok": True,
        "emotion": emotion,
        "song": song_links.get(emotion, ""),
        "background": background_colors.get(emotion, "#ffffff"),
        "robot_action": EMOTION_TO_POSE.get(emotion, "not_mapped")
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
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").lower().strip()

    if not text:
        return jsonify({"ok": False, "error": "No text provided"})

    emotion = "neutral"
    # 1. Try explicit keywords first
    if any(word in text for word in ["happy", "joy", "great", "good", "awesome", "fantastic"]):
        emotion = "happy"
    elif any(word in text for word in ["angry", "mad", "furious", "hate", "terrible", "frustrated"]):
        emotion = "angry"
    elif any(word in text for word in ["sad", "cry", "depressed", "unhappy", "bad"]):
        emotion = "sad"
    else:
        # 2. Fallback to smarter sentiment analysis
        try:
            from textblob import TextBlob
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.15:
                emotion = "happy"
            elif polarity < -0.15:
                emotion = "sad" # Default negative to sad
        except ImportError:
            pass

    lock_emotion_and_trigger(emotion)
    
    return jsonify({"ok": True, "emotion": emotion, "text": text})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False, threaded=True)