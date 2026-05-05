"""Constants used throughout the project."""

from pathlib import Path

# Project root and data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"

CACHE_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# BP4D data constants
# Raw sequences and AU coding live in the shared group folder on HPC;
# fall back to the local data dir otherwise.
HPC_DATA_DIR = Path("/home/semedit/data")

BP4D_SEQUENCES_DIR = (
    HPC_DATA_DIR / "BP4D" / "Sequences"
    if HPC_DATA_DIR.exists()
    else DATA_DIR / "BP4D" / "Sample" / "Sequences"
)
BP4D_CODING_DIR = (
    HPC_DATA_DIR / "BP4D" / "AUCoding"
    if HPC_DATA_DIR.exists()
    else DATA_DIR / "BP4D" / "AUCoding"
)
BP4D_DATA_DIR = (
    DATA_DIR / "BP4D" if HPC_DATA_DIR.exists() else DATA_DIR / "BP4D" / "Sample"
)
BP4D_INDEX_PATH = BP4D_DATA_DIR / "bp4d_index.parquet"
BP4D_AU_DISTANCES_PATH = BP4D_DATA_DIR / "au_distances.h5"
BP4D_VAE_LATENTS_PATH = DATA_DIR / "BP4D" / "vae_latents.h5"

# On HPC use the proper split indices; locally fall back to the single sample index.
BP4D_TRAIN_INDEX_PATH = (
    BP4D_INDEX_PATH.with_stem("bp4d_train_index")
    if HPC_DATA_DIR.exists()
    else BP4D_INDEX_PATH
)
BP4D_VAL_INDEX_PATH = (
    BP4D_INDEX_PATH.with_stem("bp4d_val_index")
    if HPC_DATA_DIR.exists()
    else BP4D_INDEX_PATH
)
BP4D_TEST_INDEX_PATH = (
    BP4D_INDEX_PATH.with_stem("bp4d_test_index")
    if HPC_DATA_DIR.exists()
    else BP4D_INDEX_PATH
)

BP4D_PREPROCESSED_DIR = BP4D_DATA_DIR / "Preprocessed"
BP4D_EMBEDDINGS_DIR = BP4D_DATA_DIR / "Embeddings"

BP4D_DATA_DIR.mkdir(exist_ok=True, parents=True)
BP4D_PREPROCESSED_DIR.mkdir(exist_ok=True, parents=True)
BP4D_EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)

# AU columns coded in BP4D, zero-padded to match FACS convention
BP4D_AU_COLUMNS = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU09",
    "AU10",
    "AU11",
    "AU12",
    "AU13",
    "AU14",
    "AU15",
    "AU16",
    "AU17",
    "AU18",
    "AU19",
    "AU20",
    "AU22",
    "AU23",
    "AU24",
    "AU27",
    "AU28",
]

# Subset of BP4D_AU_COLUMNS that also have intensity coding (0-5 scale)
BP4D_AU_INTENSITY_COLUMNS = ["AU06", "AU10", "AU12", "AU14", "AU17"]

# AUs with occurrence coding only (no intensity)
BP4D_AU_OCCURRENCE_COLUMNS = [
    au for au in BP4D_AU_COLUMNS if au not in BP4D_AU_INTENSITY_COLUMNS
]

# Maps each AU name to its index column in the parquet — intensity AUs use the
# _int column (0-5 scale) rather than the coarser occurrence column (0/1).
BP4D_AU_COLUMN_MAP: dict[str, str] = {
    au: (f"{au}_int" if au in BP4D_AU_INTENSITY_COLUMNS else au)
    for au in BP4D_AU_COLUMNS
}

# 001 up to 023
BP4D_FEMALE_SUBJECTS = list(f"F{str(i).zfill(3)}" for i in range(1, 24))

# 001 up to 018
BP4D_MALE_SUBJECTS = list(f"M{str(i).zfill(3)}" for i in range(1, 19))

BP4D_SUBJECTS = BP4D_FEMALE_SUBJECTS + BP4D_MALE_SUBJECTS

# Action Unit constants
AU_TO_FACS_MAP = {
    "AU01": "Inner Brow Raiser",
    "AU02": "Outer Brow Raiser",
    "AU03": "Inner corner Brow Tightener",
    "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser",
    "AU07": "Lid Tightener",
    "AU08": "Lips Toward Each Other",
    "AU09": "Nose Wrinkler",
    "AU10": "Upper Lip Raiser",
    "AU11": "Nasolabial Deepener",
    "AU12": "Lip Corner Puller",
    "AU13": "Sharp Lip Puller",
    "AU14": "Dimpler",
    "AU15": "Lip Corner Depressor",
    "AU16": "Lower Lip Depressor",
    "AU17": "Chin Raiser",
    "AU18": "Lip Pucker",
    "AU19": "Tongue Show",
    "AU20": "Lip Stretcher",
    "AU21": "Neck Tightener",
    "AU22": "Lip Funneler",
    "AU23": "Lip Tightener",
    "AU24": "Lip Pressor",
    "AU25": "Lip Part",
    "AU26": "Jaw Drop",
    "AU27": "Mouth Stretch",
    "AU28": "Lip Suck",
    "AU29": "Jaw Thrust",
    "AU30": "Jaw Sideways",
    "AU31": "Jaw Clencher",
    "AU32": "Lip Bite",
    "AU33": "Cheek Blow",
    "AU34": "Cheek Puff",
    "AU35": "Cheek Suck",
    "AU37": "Lip Wipe",
    "AU38": "Nostril Dilator",
    "AU39": "Nostril Compressor",
    "AU40": "Sniff",
    "AU41": "Lid Drop",
    "AU42": "Slit",
    "AU43": "Eyes Closed",
    "AU44": "Squint",
    "AU45": "Blink",
    "AU46": "Wink",
    "AU51": "Head turn left",
    "AU52": "Head turn right",
    "AU53": "Head up",
    "AU54": "Head down",
    "AU55": "Head tilt left",
    "AU56": "Head tilt right",
    "AU57": "Head forward",
    "AU58": "Head backward",
    "AU61": "Eyes turn left",
    "AU62": "Eyes turn right",
    "AU63": "Eyes up",
    "AU64": "Eyes down",
    "AU65": "Strabismus",
    "AU66": "Cross-eyed",
}

AU_TO_EMOTION_MAP = {
    "AU1": ["sadness", "surprise", "fear"],
    "AU2": ["surprise", "fear"],
    "AU4": ["sadness", "fear", "anger"],
    "AU5": ["surprise", "fear", "anger"],
    "AU6": ["happiness", "disgust", "contempt"],
    "AU7": ["fear", "anger"],
    "AU8": ["none"],
    "AU9": ["disgust"],
    "AU11": ["disgust", "fear"],
    "AU12": ["happiness", "contempt"],
    "AU14": ["contempt"],
    "AU15": ["sadness", "disgust"],
    "AU17": ["disgust"],
    "AU20": ["fear"],
    "AU23": ["anger"],
    "AU25": ["happiness", "surprise", "fear"],
    "AU26": ["fear", "surprise"],
    "AU38": ["anger"],
}


RANDOM_SEED = 4040
