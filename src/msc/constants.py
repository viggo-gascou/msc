"""Constants used throughout the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"

DEEP_FACE_MODELS = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet",
    "Buffalo_L",
]
DEEP_FACE_DETECTORS = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "fastmtcnn",
    "retinaface",
    "mediapipe",
    "yolov8n",
    "yolov8m",
    "yolov8l",
    "yolov11n",
    "yolov11s",
    "yolov11m",
    "yolov11l",
    "yolov12n",
    "yolov12s",
    "yolov12m",
    "yolov12l",
    "yunet",
    "centerface",
]
