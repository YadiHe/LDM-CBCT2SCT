"""HU clip range and (de)normalization helpers.

Single source of truth for CLIP_MIN / CLIP_MAX. Bumping these values
requires regenerating data/preprocessed and the manifest — the saved
preprocess_metadata.json carries the same range, and `validate_manifest_clip`
will fail-fast if a stale manifest is paired with new code.

Functions are pure arithmetic so they work transparently on torch tensors,
numpy arrays, and python scalars.
"""

CLIP_MIN = -1024.0
CLIP_MAX = 2000.0


def to_hu(x_norm):
    """Map normalized [-1, 1] back to HU using the global CLIP range."""
    return ((x_norm + 1.0) * 0.5 * (CLIP_MAX - CLIP_MIN)) + CLIP_MIN


def from_hu(x_hu):
    """Map HU to normalized [-1, 1] using the global CLIP range (no clipping)."""
    return ((x_hu - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)) * 2.0 - 1.0
