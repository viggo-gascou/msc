"""BP4D dataset utilities — index loading and frame path resolution."""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..constants import BP4D_INDEX_PATH


def load_index() -> pd.DataFrame:
    """Load the BP4D coded-frame index.

    Returns:
        DataFrame with columns: subject, task, frame, AU*, AU*_int.

    Raises:
        FileNotFoundError: If the index parquet has not been built yet.
    """
    if not BP4D_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found: {BP4D_INDEX_PATH} — run build_bp4d_index.py first"
        )
    return pd.read_parquet(BP4D_INDEX_PATH)


def resolve_frame_path(
    root: Path, subject: str, task: str, img_frame: int
) -> Path | None:
    """Resolve the image path for a single frame, handling mixed padding conventions.

    BP4D images use a mix of 4-digit padding, 3-digit or 2-digit padding, depending
    on the subject/task.

    Args:
        root:
            Root sequences directory (e.g. ~/projects/semedit/data/BP4D/Sequences).
        subject:
            Subject ID (e.g. 'F001').
        task:
            Task ID (e.g. 'T1').
        img_frame:
            0-based image frame number (AU frame number - 1).

    Returns:
        Path to the image if found, None otherwise.
    """
    for digits in (4, 3, 2):
        path = root / subject / task / f"{img_frame:0{digits}d}.jpg"
        if path.exists():
            return path
    return None


def coded_frame_paths(
    index: pd.DataFrame, root: Path
) -> dict[str, dict[str, list[Path]]]:
    """Return {task: {subject: [img_path, ...]}} for all coded frames in the index.

    AU coding uses 1-based frame numbers; image filenames are 0-based (frame N-1).
    Frames whose image file cannot be found under any padding convention are skipped.

    Args:
        index:
            DataFrame as returned by load_index().
        root:
            Root sequences directory.

    Returns:
        Nested dict mapping task → subject → sorted list of image paths.
    """
    tasks: dict[str, dict[str, list[Path]]] = defaultdict(dict)
    for (subject, task), group in index.groupby(["subject", "task"]):
        paths = []
        for au_frame in group["frame"]:
            path = resolve_frame_path(root, subject, task, int(au_frame) - 1)
            if path is not None:
                paths.append(path)
        if paths:
            tasks[task][subject] = sorted(paths)
    return dict(tasks)
