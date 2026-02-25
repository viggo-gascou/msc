"""Enums for the project."""

from enum import Enum, StrEnum, auto


class LowStrEnum(str, Enum):
    """StrEnum where auto() returns the field name in lower case."""

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list
    ) -> str:
        return name.lower()

    def __str__(self) -> str:
        """Return the value in upper case for better readability."""
        return self.value.upper()

    def __repr__(self) -> str:
        """Return the value in upper case for better readability."""
        return self.value.upper()


class LogLevel(StrEnum):
    """Logging levels."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class IdentityBackend(LowStrEnum):
    """Identity Loss Backend Detectors."""

    OPENCV = "opencv"
    SSD = "ssd"
    DLIB = "dlib"
    MTCNN = "mtcnn"
    FAST_MTCNN = "fastmtcnn"
    RETINAFACE = "retinaface"
    MEDIAPIPE = "mediapipe"
    YOLO_V8N = "yolov8n"
    YOLO_V8M = "yolov8m"
    YOLO_V8L = "yolov8l"
    YOLO_V11N = "yolov11n"
    YOLO_V11S = "yolov11s"
    YOLO_V11M = "yolov11m"
    YOLO_V11L = "yolov11l"
    YOLO_V12N = "yolov12n"
    YOLO_V12S = "yolov12s"
    YOLO_V12M = "yolov12m"
    YOLO_V12L = "yolov12l"
    YUNET = "yunet"
    CENTERFACE = "centerface"


class IdentityModel(LowStrEnum):
    """Identity model types."""

    VGG_FACE = "VGG-Face"
    FACENET = "Facenet"
    FACENET512 = "Facenet512"
    OPEN_FACE = "OpenFace"
    DEEP_FACE = "DeepFace"
    DEEP_ID = "DeepID"
    ARC_FACE = "ArcFace"
    D_LIB = "Dlib"
    S_FACE = "SFace"
    GHOST_FACE_NET = "GhostFaceNet"
    BUFFALO_L = "Buffalo_L"
