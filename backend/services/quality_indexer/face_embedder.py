"""
Face embedder — ArcFace evidence from frames already in RAM.

The frames arrive straight from the exporter's write loop (the 256×256 face
crops sampled while face_crop/{segment_id}.mp4 was being written), so this
module never opens a video file: no re-decode of an output we just encoded,
no keyframe seeks, no cv2 at all. What remains is the useful work only —
buffalo_l inference (detection for the 5-point alignment + ArcFace embedding
+ gender/age heads) on a handful of frames per segment, on CPU.

The only file in the service that touches the insightface model.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from services.quality_indexer.identity_records import (
    SegmentIdentityRecord,
    build_identity_record,
)


class FaceEmbedder:
    """Lazy buffalo_l wrapper: BGR frames in → SegmentIdentityRecord out."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._face_app = None

    def _load(self):
        if self._face_app is not None:
            return
        from insightface.app import FaceAnalysis

        models_root = self.models_dir / "insightface"
        self._face_app = FaceAnalysis(
            name="buffalo_l", root=str(models_root),
            providers=["CPUExecutionProvider"],
        )
        # det_size modest — input is a 256×256 face crop, not a full scene
        self._face_app.prepare(ctx_id=-1, det_size=(256, 256))
        logger.info("insightface buffalo_l loaded for speaker identity (CPU)")

    def embed_frames(
        self, frames: List[np.ndarray]
    ) -> Optional[SegmentIdentityRecord]:
        """Collect ArcFace embedding + gender/age evidence from BGR frames.

        Returns None when no face was detected in any frame (or no frames
        were provided) — the segment simply contributes no identity evidence.
        """
        if not frames:
            return None
        self._load()

        embeddings, genders, ages = [], [], []
        for frame in frames:
            if frame is None or frame.size == 0:
                continue
            faces = self._face_app.get(frame)
            if not faces:
                continue
            # Largest face — the crop is speaker-centered by construction
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            if face.normed_embedding is not None:
                embeddings.append(np.asarray(face.normed_embedding, dtype=np.float32))
            gender = getattr(face, "gender", None)
            age = getattr(face, "age", None)
            if gender is not None:
                genders.append(int(gender))
            if age is not None:
                ages.append(float(age))

        return build_identity_record(embeddings, genders, ages)
