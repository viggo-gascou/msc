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


class ONNXProvider(str, Enum):
    """ONNX Runtime execution providers."""

    CUDA = "CUDAExecutionProvider"
    CPU = "CPUExecutionProvider"
    COREML = "CoreMLExecutionProvider"


class IdentityModel(str, Enum):
    """InsightFace model packs.

    See https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo
    for model descriptions.
    """

    BUFFALO_L = "buffalo_l"
    BUFFALO_M = "buffalo_m"
    BUFFALO_S = "buffalo_s"
    BUFFALO_SC = "buffalo_sc"
    ANTELOPE_V2 = "antelopev2"
