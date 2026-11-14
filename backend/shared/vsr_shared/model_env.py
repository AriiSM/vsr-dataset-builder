"""
Model cache redirection — ONE portable folder for every model.

Whisper, Silero, insightface and HF-hub models normally scatter their
caches across the user's home directory. Calling apply_model_env() BEFORE
any ML import points all of them under models_dir:

    models/
    ├── talknet_asd.pth · syncnet_v2.pth     (manifest: direct/manual)
    ├── huggingface/                         (HF_HOME → whisper, pyannote)
    ├── torch/                               (TORCH_HOME → silero, retinaface)
    └── insightface/                         (buffalo_l)

Result: models/ is complete and portable — copy it to an offline machine
and nothing re-downloads. Explicit user-set env vars are respected.
"""

import os
from pathlib import Path


def apply_model_env(models_dir: Path) -> dict:
    """Point every model cache under models_dir (no override of user env).

    Returns the mapping actually in effect (for doctor's report)."""
    models_dir = Path(models_dir).resolve()
    defaults = {
        "TORCH_HOME": str(models_dir / "torch"),
        "HF_HOME": str(models_dir / "huggingface"),
        "INSIGHTFACE_HOME": str(models_dir / "insightface"),
    }
    applied = {}
    for key, value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
        applied[key] = os.environ[key]
    return applied
