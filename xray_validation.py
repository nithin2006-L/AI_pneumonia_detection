"""
xray_validation.py
==================
Gatekeeper that decides whether an uploaded file is a frontal chest radiograph
BEFORE it is allowed to reach the pneumonia model.

Pipeline (every stage can reject the upload with a human-readable reason):

  Stage 1  Decode            - file really is an image / DICOM, not renamed junk
  Stage 2  DICOM metadata    - (DICOM only) Modality / BodyPart must not contradict "chest X-ray"
  Stage 3  Basic quality     - minimum size, sane aspect ratio, not blank / flat
  Stage 4  Not colour/graphic- rejects colour photos, screenshots, documents, charts
  Stage 5  Chest structure   - soft checks (need >= MIN_SOFT_PASSES of 4):
                               left/right symmetry, bright mediastinum between darker
                               lung fields, smooth radiographic texture, dark-air regions

Public API
----------
    result = validate_chest_xray(file_bytes, filename)
    if not result.is_valid:
        flash(result.message)          # show to the user
    result.pil_image                   # decoded RGB PIL image (when decoding worked)
    result.details                     # dict of measured values (for logs / tuning)

NOTE: these are image-statistics heuristics, not a trained classifier. They are tuned
to be conservative (avoid rejecting genuine radiographs). Tune the THRESHOLDS block
below using a handful of your own valid / invalid samples. For maximum robustness you
can later add a small CXR-vs-not-CXR CNN as a Stage 6 (see bottom of this file).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# THRESHOLDS  (tune these against your own data)
# ----------------------------------------------------------------------------
MIN_SIDE_PX = 128                 # smaller images are rejected
ASPECT_RATIO_RANGE = (0.55, 1.85)  # width / height of a frontal chest film
MIN_DYNAMIC_RANGE = 50            # p99 - p1 grey levels (0-255); blank/flat images fail
MIN_UNIQUE_LEVELS = 40            # graphics / screenshots have very few grey levels
MAX_WHITE_FRACTION = 0.40         # documents & screenshots are mostly pure white
MAX_MEAN_SATURATION = 85.0        # colour-photo rejection (matches your original rule)
MAX_CHANNEL_STD = 35.0
MIN_SOFT_PASSES = 2               # how many of the 4 chest-structure checks must pass

SYMMETRY_MIN = 0.30               # left vs mirrored-right correlation
MEDIASTINUM_CONTRAST_MIN = 0.02   # centre column brighter than lung zones (0-1 scale)
MAX_EDGE_DENSITY = 0.12           # fraction of strong edges (graphics have many)
MIN_DARK_FRACTION = 0.03          # share of "air-dark" pixels (lungs / background)

ALLOWED_DICOM_MODALITIES = {"CR", "DX", "DR", "RG", "XA", "OT", ""}
CHEST_BODY_PARTS = ("CHEST", "THORAX", "LUNG", "THORACIC", "CXR", "RIB", "HEART")


@dataclass
class ValidationResult:
    is_valid: bool
    message: str = ""
    stage: str = ""                       # which stage rejected / "passed"
    pil_image: Image.Image | None = None  # decoded RGB image (None if decode failed)
    details: dict = field(default_factory=dict)


def _reject(stage, message, pil=None, **details):
    return ValidationResult(False, message, stage, pil, details)


# ----------------------------------------------------------------------------
# Stage 1 / 2 : decoding
# ----------------------------------------------------------------------------
def _decode_dicom(file_bytes):
    """Return (PIL RGB image, metadata-problem-or-None)."""
    import pydicom  # imported lazily so the module loads even without pydicom

    ds = pydicom.dcmread(io.BytesIO(file_bytes), force=False)
    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM file does not contain any image data.")

    # ---- Stage 2: metadata sanity (only reject on a clear contradiction) ----
    modality = str(getattr(ds, "Modality", "")).strip().upper()
    if modality not in ALLOWED_DICOM_MODALITIES:
        return None, (f"This DICOM is a '{modality}' study, not a chest radiograph "
                      f"(expected CR/DX).")
    body_part = str(getattr(ds, "BodyPartExamined", "")).strip().upper()
    if body_part and not any(k in body_part for k in CHEST_BODY_PARTS):
        return None, (f"This DICOM is labelled body part '{body_part.title()}', "
                      f"not chest.")

    arr = ds.pixel_array
    if arr.ndim != 2:
        raise ValueError("Only single-frame, greyscale DICOM radiographs are supported.")

    img = arr.astype(np.float32)
    img = img * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        img = np.max(img) - img
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        raise ValueError("DICOM pixel data has no valid values.")
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        raise ValueError("DICOM image has no intensity variation.")
    img8 = (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img8, mode="L").convert("RGB"), None


def _decode_image(file_bytes):
    """Decode PNG/JPG bytes; verify() catches truncated or fake files."""
    try:
        Image.open(io.BytesIO(file_bytes)).verify()
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Stage 3-5 : pixel analysis
# ----------------------------------------------------------------------------
def _analyse(pil_rgb, is_dicom):
    """Return (rejection_message_or_None, details)."""
    rgb = np.array(pil_rgb)
    h, w = rgb.shape[:2]
    d = {"width": w, "height": h}

    # ---- Stage 3: basic quality ------------------------------------------
    if min(h, w) < MIN_SIDE_PX:
        return (f"Image is too small ({w}x{h}px). A chest X-ray should be at least "
                f"{MIN_SIDE_PX}px on its shortest side."), d
    aspect = w / h
    d["aspect_ratio"] = round(aspect, 3)
    if not (ASPECT_RATIO_RANGE[0] <= aspect <= ASPECT_RATIO_RANGE[1]):
        return "The image proportions are unusual for a chest X-ray (extremely wide or tall).", d

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    p1, p99 = np.percentile(gray, [1, 99])
    d["dynamic_range"] = float(p99 - p1)
    if p99 - p1 < MIN_DYNAMIC_RANGE:
        return "The image is blank or has almost no contrast - it is not a chest X-ray.", d

    # ---- Stage 4: reject colour photos / graphics / documents --------------
    if not is_dicom:  # DICOM is already normalised to grey
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mean_sat = float(hsv[:, :, 1].mean())
        ch_std = float(np.std([rgb[:, :, c].mean() for c in range(3)]))
        d["mean_saturation"], d["channel_std"] = round(mean_sat, 1), round(ch_std, 1)
        # Blue/green-tinted radiographs have low saturation spread; real colour
        # photos are strongly saturated AND have unbalanced channels.
        if mean_sat > MAX_MEAN_SATURATION and ch_std > MAX_CHANNEL_STD:
            return "This looks like a colour photograph, not a chest X-ray.", d

    unique_levels = int(len(np.unique(gray)))
    white_fraction = float((gray >= 245).mean())
    d["unique_levels"], d["white_fraction"] = unique_levels, round(white_fraction, 3)
    if unique_levels < MIN_UNIQUE_LEVELS:
        return "The image has very few grey levels (looks like a graphic or scanned text).", d
    if white_fraction > MAX_WHITE_FRACTION:
        return "The image is mostly white (looks like a document or screenshot).", d

    # ---- Stage 5: chest-specific structure (soft checks) -------------------
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    norm = np.clip((small.astype(np.float32) - p1) / max(p99 - p1, 1.0), 0, 1)
    blur = cv2.GaussianBlur(norm, (0, 0), 5)                     # coarse anatomy only

    # (a) left/right symmetry
    left = blur[:, :128]
    right = np.fliplr(blur[:, 128:])
    lv, rv = left.flatten() - left.mean(), right.flatten() - right.mean()
    denom = np.linalg.norm(lv) * np.linalg.norm(rv)
    symmetry = float(np.dot(lv, rv) / denom) if denom > 1e-6 else 0.0
    d["symmetry"] = round(symmetry, 3)

    # (b) bright mediastinum/spine between darker lung fields
    centre = blur[50:200, 112:144].mean()
    lungs = np.concatenate([blur[50:200, 35:105].flatten(), blur[50:200, 151:221].flatten()]).mean()
    mediastinum_contrast = float(centre - lungs)
    d["mediastinum_contrast"] = round(mediastinum_contrast, 3)

    # (c) smooth radiographic texture (graphics / drawings have many hard edges)
    edges = cv2.Canny(small, 100, 200)
    edge_density = float((edges > 0).mean())
    d["edge_density"] = round(edge_density, 3)

    # (d) presence of air-dark regions (lungs / background)
    dark_fraction = float((norm < 0.25).mean())
    d["dark_fraction"] = round(dark_fraction, 3)

    checks = {
        "symmetry": symmetry >= SYMMETRY_MIN,
        "mediastinum": mediastinum_contrast >= MEDIASTINUM_CONTRAST_MIN,
        "texture": edge_density <= MAX_EDGE_DENSITY,
        "air_regions": dark_fraction >= MIN_DARK_FRACTION,
    }
    d["soft_checks"] = checks
    d["soft_passes"] = sum(checks.values())
    if d["soft_passes"] < MIN_SOFT_PASSES:
        return ("The image does not show the typical structure of a frontal chest X-ray "
                "(symmetric lung fields either side of the mediastinum)."), d

    return None, d


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def validate_chest_xray(file_bytes: bytes, filename: str) -> ValidationResult:
    """Validate an uploaded file. Never raises - always returns a ValidationResult."""
    ext = os.path.splitext(filename or "")[1].lower()
    is_dicom = ext == ".dcm"

    if not file_bytes:
        return _reject("decode", "The uploaded file is empty.")

    # Stage 1 + 2
    try:
        if is_dicom:
            pil, meta_problem = _decode_dicom(file_bytes)
            if meta_problem:
                return _reject("dicom_metadata", meta_problem)
        else:
            pil = _decode_image(file_bytes)
    except Exception as exc:
        return _reject("decode", f"The file could not be read as a valid DICOM image: {exc}")

    if pil is None:
        return _reject("decode", "The file is corrupted or is not a real PNG/JPG image.")

    # Stage 3-5
    try:
        problem, details = _analyse(pil, is_dicom)
    except Exception as exc:  # fail closed: if analysis breaks, do not let it through
        return _reject("analysis", f"The image could not be analysed ({exc}).", pil)

    if problem:
        return _reject("pixel_analysis", problem, pil, **details)

    return ValidationResult(True, "", "passed", pil, details)


# ----------------------------------------------------------------------------
# Optional Stage 6 (recommended for production): a tiny CXR / not-CXR classifier.
# Train a 2-class MobileNet on a few thousand chest X-rays vs random images
# (e.g. CIFAR / ImageNet samples), then call it here and reject when
# P(chest_xray) < 0.5. The heuristics above remain as a cheap first filter.
# ----------------------------------------------------------------------------