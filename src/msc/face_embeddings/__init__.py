"""Face embedding models for identity conditioning and loss computation."""

import warnings

from .adaface import AdaFaceEmbedding, load_adaface
from .arcface import ArcFaceEmbedding
from .base import FaceEmbedding
from .preprocessor import FacePreprocessor

# insightface calls skimage.transform.estimate (deprecated since skimage 0.26)
warnings.filterwarnings(
    "ignore",
    message=r".*`estimate` is deprecated.*",
    category=FutureWarning,
    module="insightface",
)
