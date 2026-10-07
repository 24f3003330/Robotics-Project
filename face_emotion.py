"""
face_emotion.py - facial emotion recognition for the /camera page.

    webcam frame (BGR)
        -> YuNet face detector        cv2.FaceDetectorYN, run on a 640 px wide copy
        -> largest face only          one person in front of the robot
        -> FER+ classifier            cv2.dnn, 64x64 grayscale -> 8 emotion scores
        -> softmax                    probabilities per emotion

Both models are small ONNX files run by OpenCV itself, so no extra Python
package is needed. They are downloaded once into models/face/ (checksums
verified) and loaded once per process. If they cannot be obtained, the old
Haar smile detector (happy / neutral only) is used instead and the reason is
reported, so the camera page still works.

This module knows nothing about fusion, music or the robot; app.py decides
what a prediction does.
"""

import hashlib
import os
import threading
import traceback
import urllib.request

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face")

# name -> (url, sha256)
MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "emotion-ferplus-8.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
        "emotion_ferplus/model/emotion-ferplus-8.onnx",
        "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c",
    ),
}

# FER+ output order (from the model's documentation), renamed to the app's
# emotion names so music/background/robot mappings apply unchanged.
FERPLUS_LABELS = ("neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt")

DETECT_WIDTH = 640               # faces are found on a downscaled copy (fast at 1080p)
DETECT_SCORE_THRESHOLD = 0.8     # YuNet face confidence
MIN_FACE_FRACTION = 0.06         # ignore "faces" smaller than this share of the frame height


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_file(name, log):
    url, expected = MODEL_FILES[name]
    path = os.path.join(MODEL_DIR, name)
    if os.path.isfile(path) and _sha256(path) == expected:
        return path
    os.makedirs(MODEL_DIR, exist_ok=True)
    log(f"Downloading {name} (first run needs internet) ...")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    actual = _sha256(tmp)
    if actual != expected:
        os.remove(tmp)
        raise RuntimeError(f"{name} failed its checksum (got {actual[:12]}..., "
                           f"expected {expected[:12]}...)")
    os.replace(tmp, path)
    return path


class FaceEmotionModel:
    """Loaded once; analyze() is thread-safe (the OpenCV nets are not)."""

    def __init__(self, log=print):
        self._log = log
        self._lock = threading.Lock()
        self._loaded = False
        self._detector = None
        self._classifier = None
        self._haar_face = None
        self._haar_smile = None
        self.kind = None          # "ferplus" | "haar-smile"
        self.labels = FERPLUS_LABELS
        self.warning = None

    def ensure_loaded(self):
        """Load (downloading if needed) exactly once. Never raises."""
        with self._lock:
            if self._loaded:
                return
            try:
                detector_path = _ensure_file("face_detection_yunet_2023mar.onnx", self._log)
                classifier_path = _ensure_file("emotion-ferplus-8.onnx", self._log)
                self._detector = cv2.FaceDetectorYN.create(
                    detector_path, "", (DETECT_WIDTH, DETECT_WIDTH), DETECT_SCORE_THRESHOLD, 0.3, 50)
                self._classifier = cv2.dnn.readNetFromONNX(classifier_path)
                self.kind = "ferplus"
                self.labels = FERPLUS_LABELS
                self._log("Facial emotion model ready: YuNet face detector + FER+ "
                          f"({', '.join(FERPLUS_LABELS)})")
            except Exception as exc:
                traceback.print_exc()
                self.warning = (f"Facial emotion model unavailable ({exc}); falling back to smile "
                                f"detection, which can only tell happy from neutral.")
                self._log(f"WARNING: {self.warning}")
                self._haar_face = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                self._haar_smile = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_smile.xml")
                self.kind = "haar-smile"
                self.labels = ("neutral", "happy")
            self._loaded = True

    def analyze(self, frame):
        """frame: BGR image -> None (no face) or
        {"box": (x, y, w, h), "probabilities": {emotion: p}, "emotion": str, "confidence": float}
        for the largest face."""
        self.ensure_loaded()
        with self._lock:
            if self.kind == "ferplus":
                return self._analyze_ferplus(frame)
            return self._analyze_haar(frame)

    # ------------------------------------------------------------------
    def _analyze_ferplus(self, frame):
        height, width = frame.shape[:2]
        scale = DETECT_WIDTH / float(width) if width > DETECT_WIDTH else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._detector.detect(small)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = (int(round(v / scale)) for v in face[:4])
        if h < MIN_FACE_FRACTION * height:
            return None
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        blob = cv2.resize(gray, (64, 64)).astype(np.float32).reshape(1, 1, 64, 64)
        self._classifier.setInput(blob)
        scores = self._classifier.forward()[0].astype(np.float64)
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        probabilities = {label: float(p) for label, p in zip(FERPLUS_LABELS, probs)}
        emotion = max(probabilities, key=probabilities.get)
        return {"box": (x0, y0, x1 - x0, y1 - y0), "probabilities": probabilities,
                "emotion": emotion, "confidence": probabilities[emotion],
                "detector_score": float(face[-1])}

    def _analyze_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                 minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[y:y + h, x:x + w]
        if face_gray.size == 0:
            return None
        smiles = self._haar_smile.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=18,
                                                   minSize=(25, 25))
        emotion = "happy" if len(smiles) > 0 else "neutral"
        probabilities = {"neutral": 0.0, "happy": 0.0}
        probabilities[emotion] = 1.0
        return {"box": (int(x), int(y), int(w), int(h)), "probabilities": probabilities,
                "emotion": emotion, "confidence": 1.0, "detector_score": None}
