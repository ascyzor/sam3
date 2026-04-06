"""
SAM3 Scene Auto-Segmenter
==========================
Two-stage pipeline for automatic segmentation with consistent cross-frame object IDs.

  Stage 1 — Vocabulary Discovery  (Florence-2-base, ~230 MB)
      Samples every N frames from the input sequence and runs Florence-2's
      zero-prompt object-detection task (<OD>) to find what objects are
      present in the scene.  The discovered labels are formatted into a
      SAM3-compatible period-separated text prompt.

  Stage 2 — Video Segmentation  (SAM3 / SAM3.1 video predictor)
      Uses the discovered prompt (or an explicit --prompt if provided) with
      the SAM3 video predictor and memory tracker so the same physical object
      keeps the SAME ID across all frames.

Output (per frame)
------------------
  <stem>_mask.png  – 16-bit grayscale PNG, pixel value = object ID (0 = background)
  <stem>_info.txt  – TSV: id | score | x0 | y0 | x1 | y1  (absolute px)

Usage
-----
  # Fully automatic (Florence-2 discovers vocabulary):
  python sam3_scene_segment.py \\
      --input_dir /path/to/frames --output_dir /out

  # Explicit prompt (skip Florence-2):
  python sam3_scene_segment.py \\
      --input_dir /path/to/frames --output_dir /out \\
      --prompt "person, car, bicycle"

  # From a video file:
  python sam3_scene_segment.py \\
      --input_video clip.mp4 --output_dir /out

Requirements
------------
  pip install -e ".[notebooks]"
  pip install "transformers>=4.38"   # for Florence-2
  pip install opencv-python          # only needed for --input_video
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

_FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"

_LOCAL_CKPT_CANDIDATES = {
    "sam3":   ["ckpt/sam3.pt",             "checkpoints/sam3.pt"],
    "sam3.1": ["ckpt/sam3.1_multiplex.pt", "checkpoints/sam3.1_multiplex.pt"],
}


# ---------------------------------------------------------------------------
# Checkpoint resolver  (mirrors sam3_auto_segment.py)
# ---------------------------------------------------------------------------

def _resolve_checkpoint(checkpoint_path, version):
    """Return (resolved_path, load_from_HF)."""
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.is_absolute():
            p = _SCRIPT_DIR / p
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return str(p), False

    for rel in _LOCAL_CKPT_CANDIDATES.get(version, []):
        p = _SCRIPT_DIR / rel
        if p.exists():
            print(f"    Using local checkpoint: {p}")
            return str(p), False

    print("    No local checkpoint found – will auto-download from HuggingFace.")
    return None, True


# ---------------------------------------------------------------------------
# Shared rasterisation / IO helpers
# ---------------------------------------------------------------------------

def _build_id_image(obj_ids, binary_masks, height, width, non_overlap=False):
    """
    Paint (obj_id, mask) pairs into a uint16 image.

    non_overlap=False (default): first entry in list wins on overlap.
    non_overlap=True:            once a pixel is claimed it cannot be overwritten.
    """
    id_image = np.zeros((height, width), dtype=np.uint16)
    if len(obj_ids) == 0:
        return id_image

    masks_np = np.asarray(binary_masks, dtype=bool)
    ids_np   = np.asarray(obj_ids,      dtype=np.int32)

    if non_overlap:
        for obj_id, mask in zip(ids_np, masks_np):
            # +1 so SAM3's 0-based IDs never collide with background (0)
            id_image[mask & (id_image == 0)] = int(obj_id) + 1
    else:
        # Write in reverse so first entry (index 0) wins on overlap.
        for obj_id, mask in zip(ids_np[::-1], masks_np[::-1]):
            id_image[mask] = int(obj_id) + 1  # +1: reserve 0 for background

    return id_image


def _id_image_to_color(id_image):
    """
    Convert a uint16 ID image to an RGB color image for visualization.

    Each unique object ID receives a distinct, deterministic color derived
    from the golden-ratio HSV sequence.  ID 0 (background) is black.
    Returns a uint8 RGB numpy array of shape (H, W, 3).
    """
    import colorsys

    h, w = id_image.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for obj_id in np.unique(id_image):
        if obj_id == 0:
            continue  # background stays black
        # Golden-ratio hue shift (~137.5° per ID) → visually distinct colors
        hue = (int(obj_id) * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        mask = id_image == obj_id          # (H, W) bool
        color_img[mask, 0] = int(r * 255)
        color_img[mask, 1] = int(g * 255)
        color_img[mask, 2] = int(b * 255)
    return color_img


def _save_info_txt(path, obj_ids, scores, boxes_xywh_norm, orig_h, orig_w):
    """Write TSV: id | score | x0 | y0 | x1 | y1  (absolute pixel coords)."""
    lines = ["id\tscore\tx0\ty0\tx1\ty1"]
    for obj_id, score, box in zip(obj_ids, scores, boxes_xywh_norm):
        x, y, w, h = (float(v) for v in box)
        lines.append(
            f"{int(obj_id)}\t{float(score):.4f}"
            f"\t{x * orig_w:.1f}\t{y * orig_h:.1f}"
            f"\t{(x + w) * orig_w:.1f}\t{(y + h) * orig_h:.1f}"
        )
    Path(path).write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Frame-path helpers
# ---------------------------------------------------------------------------

def _sorted_frame_paths(input_dir):
    """Return image paths from a directory, sorted to match SAM3's io_utils order."""
    frames = [p for p in Path(input_dir).iterdir()
              if p.suffix.lower() in IMAGE_EXTS]
    try:
        frames.sort(key=lambda p: int(p.stem))
    except ValueError:
        frames.sort()
    return frames


def _sample_pil_images_from_dir(frame_paths, every_n):
    """Yield (index, PIL.Image) for every-N-th frame in frame_paths."""
    for i, path in enumerate(frame_paths):
        if i % every_n == 0:
            yield i, Image.open(path).convert("RGB")


def _sample_pil_images_from_video(video_path, every_n):
    """Yield (frame_index, PIL.Image) for every-N-th frame in a video file."""
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video file input.  "
            "Install with: pip install opencv-python"
        )
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            yield frame_idx, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()


# ---------------------------------------------------------------------------
# Stage 1 – Vocabulary discovery with Florence-2
# ---------------------------------------------------------------------------

def _run_florence2_od(model, processor, image, device):
    """
    Run Florence-2 <OD> on one PIL image.
    Returns a set of lowercase label strings detected in the image.
    """
    inputs = processor(
        text="<OD>",
        images=image,
        return_tensors="pt",
    ).to(device)

    # Cast pixel_values to match the model's dtype (e.g. float16 on CUDA).
    # The processor always outputs float32; mismatching with a float16 model
    # causes a RuntimeError in the vision encoder.
    model_dtype = next(model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    parsed = processor.post_process_generation(
        generated_text,
        task="<OD>",
        image_size=(image.width, image.height),
    )

    od = parsed.get("<OD>", {})
    raw_labels = od.get("labels", od.get("bboxes_labels", []))
    clean = set()
    for lbl in raw_labels:
        lbl = lbl.strip().lower()
        if not lbl:
            continue
        # Take only the first alternative when Florence outputs "trash bin/can"
        # style labels – the slash becomes a garbled merged token in SAM3's
        # CLIP tokenizer (punctuation is stripped silently, leaving "bincan").
        lbl = lbl.split("/")[0].strip()
        if lbl:
            clean.add(lbl)
    return clean


def discover_vocabulary(
    input_path,
    every_n=5,
    device=None,
    min_label_frames=1,
    min_label_freq=0.30,
    top_n=4,
):
    """
    Sample frames from *input_path* (directory or video file) and run
    Florence-2-base <OD> to discover what object categories are present.

    Parameters
    ----------
    input_path      : str – directory of frames or video file path
    every_n         : int – sample 1 frame per every_n frames (default 5)
    device          : str – "cuda", "cpu", or None (auto-select)
    min_label_frames: int   – a label must appear in at least this many sampled
                      frames to be kept (filters transient false positives)
    min_label_freq  : float – a label must appear in more than this fraction of
                      sampled frames to be kept (default 0.30 = 30 %)
    top_n           : int – after filtering, keep only the N most frequently
                      seen labels (default 4).  SAM3 uses a CLIP-style phrase
                      encoder; long prompts dilute the embedding and hurt
                      detection.  Keeping the prompt short and focused is key.

    Returns
    -------
    tuple[set[str], dict[str, int]]
        - filtered label set ready for SAM3
        - raw label_frame_count dict (all labels before any filtering), for logging
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Loading Florence-2-base on {device} …")

    # Florence-2's remote modeling_florence2.py has a hard `flash_attn` import.
    # Inject a stub so transformers' check_imports() passes, then load with
    # attn_implementation="eager" so the stub is never called at runtime.
    import importlib.machinery, sys, types
    if "flash_attn" not in sys.modules:
        try:
            import flash_attn  # noqa: F401 – use real one if available
        except ImportError:
            _stub = types.ModuleType("flash_attn")
            # __spec__ must not be None: Python 3.12's importlib.util.find_spec
            # raises ValueError when it finds a sys.modules entry with __spec__=None.
            # A ModuleSpec with loader=None makes find_spec() return the spec (not
            # None), then the metadata version-check fails (no installed package),
            # so is_flash_attn_2_available() returns False and Florence-2 skips it.
            _stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
            sys.modules["flash_attn"] = _stub

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        _FLORENCE2_MODEL_ID, trust_remote_code=True
    )
    flo_model = (
        AutoModelForCausalLM.from_pretrained(
            _FLORENCE2_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="eager",  # do not use flash_attn at runtime
        )
        .to(device)
        .eval()
    )

    # ── Sample frames and count label occurrences ─────────────────────────
    input_obj = Path(input_path)
    label_frame_count: dict[str, int] = {}
    n_sampled = 0

    if input_obj.is_dir():
        frame_paths = _sorted_frame_paths(input_path)
        if not frame_paths:
            raise RuntimeError(f"No images found in {input_path!r}")
        total = len(frame_paths)
        print(f"  {total} frames found; sampling every {every_n} → "
              f"~{(total + every_n - 1) // every_n} frames")
        image_iter = _sample_pil_images_from_dir(frame_paths, every_n)
    else:
        image_iter = _sample_pil_images_from_video(input_path, every_n)

    for frame_idx, pil_img in image_iter:
        labels = _run_florence2_od(flo_model, processor, pil_img, device)
        for lbl in labels:
            label_frame_count[lbl] = label_frame_count.get(lbl, 0) + 1
        n_sampled += 1
        if n_sampled % 10 == 0:
            print(f"    … processed {n_sampled} sampled frames", flush=True)

    print(f"  Florence-2 processed {n_sampled} sampled frame(s). "
          f"Raw label count: {len(label_frame_count)}")

    # ── Free GPU memory before loading SAM3 ───────────────────────────────
    del flo_model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Apply minimum-frame filter ─────────────────────────────────────────
    kept = {lbl: cnt for lbl, cnt in label_frame_count.items()
            if cnt >= min_label_frames}
    dropped = set(label_frame_count) - set(kept)
    if dropped:
        print(f"  Dropped {len(dropped)} label(s) seen in fewer than "
              f"{min_label_frames} frame(s): {sorted(dropped)}")

    # ── Apply frequency filter (> min_label_freq of sampled frames) ────────
    if n_sampled > 0 and min_label_freq > 0:
        freq_kept = {lbl: cnt for lbl, cnt in kept.items()
                     if cnt / n_sampled > min_label_freq}
        freq_dropped = set(kept) - set(freq_kept)
        if freq_dropped:
            print(f"  Dropped {len(freq_dropped)} label(s) below "
                  f"{min_label_freq*100:.0f}% frequency threshold: "
                  f"{sorted(freq_dropped)}")
        kept = freq_kept

    # ── Keep only the top-N most frequent labels ───────────────────────────
    # SAM3 encodes the entire prompt as ONE CLIP-style phrase embedding.
    # Too many categories dilute the embedding and push detection scores below
    # threshold.  We sort by frame-frequency and take the most prominent ones.
    if top_n and len(kept) > top_n:
        sorted_by_freq = sorted(kept.items(), key=lambda kv: kv[1], reverse=True)
        trimmed = set(lbl for lbl, _ in sorted_by_freq[:top_n])
        print(f"  Keeping top {top_n} label(s) by frequency "
              f"(dropped: {sorted(set(kept) - trimmed)})")
        kept = trimmed
    else:
        kept = set(kept)

    return kept, label_frame_count


def _deduplicate_labels(labels):
    """
    Remove labels that are word-prefix extensions of a shorter label.

    Example: {"bicycle", "bicycle wheel"} → {"bicycle"}
    because "bicycle wheel" starts with "bicycle", so the longer one is redundant.
    The rule is: if every word of label A equals the first N words of label B,
    drop B and keep A.  This prevents SAM3 from receiving competing sub-labels
    that belong to the same object category.
    """
    # Sort shortest-first so we always evaluate parent labels before children.
    by_length = sorted(labels, key=lambda l: len(l.split()))
    kept: list[str] = []
    for lbl in by_length:
        words = lbl.split()
        dominated = any(
            words[: len(k.split())] == k.split()
            for k in kept
            if len(k.split()) < len(words)
        )
        if not dominated:
            kept.append(lbl)
    return set(kept)


def labels_to_sam3_prompt(labels, label_counts=None):
    """
    Deduplicate sub-labels then format into a SAM3 comma-separated prompt.

    Labels are ordered by descending frame-frequency when label_counts is
    provided (most prominent category first), otherwise alphabetically.

    Example: {"bicycle", "bicycle wheel", "bench"}, {"bicycle":10,"bench":4}
             → "bicycle, bench"
    """
    clean = _deduplicate_labels(labels)
    if label_counts:
        ordered = sorted(clean, key=lambda l: label_counts.get(l, 0), reverse=True)
    else:
        ordered = sorted(clean)
    return ", ".join(ordered)


# ---------------------------------------------------------------------------
# Stage 2 – SAM3 video segmentation
# ---------------------------------------------------------------------------

def _run_single_label_session(predictor, input_path, label, anchor_frame,
                               threshold, frame_stems, orig_h, orig_w):
    """
    Run one complete SAM3 session for a single label.

    Returns
    -------
    dict[int, dict[int, np.ndarray]]
        frame_idx → {local_obj_id → binary_mask (H, W bool)}
    Also returns updated (orig_h, orig_w) in case they were None on entry.
    """
    response   = predictor.handle_request({
        "type":          "start_session",
        "resource_path": str(input_path),
    })
    session_id = response["session_id"]

    predictor.handle_request({
        "type":               "add_prompt",
        "session_id":         session_id,
        "frame_index":        anchor_frame,
        "text":               label,
        "output_prob_thresh": threshold,
    })

    session_masks = {}   # frame_idx → {local_obj_id → binary_mask}
    total_detections = 0

    for result in predictor.handle_stream_request({
        "type":                  "propagate_in_video",
        "session_id":            session_id,
        "propagation_direction": "both",
        "output_prob_thresh":    threshold,
    }):
        frame_idx    = result["frame_index"]
        outputs      = result["outputs"]
        obj_ids      = outputs["out_obj_ids"]
        binary_masks = outputs["out_binary_masks"]

        if orig_h is None and len(binary_masks) > 0:
            orig_h, orig_w = binary_masks.shape[-2], binary_masks.shape[-1]

        n = len(obj_ids)
        total_detections += n
        if n > 0:
            session_masks[frame_idx] = {
                int(oid): binary_masks[i].astype(bool)
                for i, oid in enumerate(obj_ids)
            }

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    print(f"    label '{label}': {total_detections} total detections across all frames")
    return session_masks, orig_h, orig_w


def segment_with_sam3(
    input_path,
    output_dir,
    prompt,
    threshold=0.3,
    anchor_frame=0,
    checkpoint_path=None,
    non_overlap=False,
    version="sam3.1",
):
    """
    Run one SAM3 session per label in *prompt*, then merge all results.

    Each label gets a clean single-word CLIP embedding → reliable detection.
    Per-session local IDs are offset into globally unique IDs before merging.
    On overlapping pixels the session with higher Florence-2 frequency (i.e.
    the label that appears first in the comma-separated prompt) wins.

    Parameters
    ----------
    input_path    : directory of frames OR path to an MP4/AVI/MOV video
    output_dir    : where to write gray_mask/ and color_mask/
    prompt        : comma-separated labels ordered by frequency descending,
                    e.g. "bicycle, bench, bird"
    threshold     : detector confidence threshold [0, 1]
    anchor_frame  : frame index where the detector fires first (default 0)
    checkpoint_path: local .pt file; None = auto-detect or HF download
    non_overlap   : if True, each pixel belongs to at most one mask
    version       : "sam3" or "sam3.1"
    """
    gray_dir  = Path(output_dir) / "gray_mask"
    color_dir = Path(output_dir) / "color_mask"
    gray_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)

    # Parse comma-separated prompt into ordered label list.
    labels = [l.strip() for l in prompt.split(",") if l.strip()]
    if not labels:
        raise ValueError(f"prompt is empty: {prompt!r}")

    # ── 1. Build SAM3 predictor (shared across all sessions) ──────────────
    print(f"  Building SAM3 {version} video predictor …")
    from sam3.model_builder import build_sam3_predictor

    resolved_ckpt, load_from_hf = _resolve_checkpoint(checkpoint_path, version)
    predictor = build_sam3_predictor(
        checkpoint_path=resolved_ckpt,
        version=version,
        use_fa3=False,
        async_loading_frames=True,
    )
    print("  Predictor ready.")

    # ── 2. Resolve frame stems and dimensions ─────────────────────────────
    input_obj = Path(input_path)
    if input_obj.is_dir():
        frame_paths = _sorted_frame_paths(input_path)
        frame_stems = [p.stem for p in frame_paths]
        with Image.open(frame_paths[0]) as _img:
            orig_w, orig_h = _img.size
    else:
        frame_stems = None
        orig_h = orig_w = None
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(str(input_obj))
            orig_w = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            _cap.release()
        except Exception:
            pass

    # ── 3. Run one session per label, collect masks in memory ─────────────
    # all_sessions: list of per-label dicts {frame_idx → {local_id → mask}}
    # ordered by label priority (first = highest Florence frequency = wins overlap)
    all_sessions = []
    for i, label in enumerate(labels):
        print(f"  [{i+1}/{len(labels)}] Running session for label: '{label}' …")
        session_masks, orig_h, orig_w = _run_single_label_session(
            predictor, input_path, label, anchor_frame,
            threshold, frame_stems, orig_h, orig_w,
        )
        all_sessions.append(session_masks)

    # ── 4. Assign globally unique IDs across sessions ─────────────────────
    # local SAM3 IDs start at 0 per session; offset so pixel 0 = background.
    # Session k's local ID j → global ID = id_offset + j + 1
    global_sessions = []   # list of {frame_idx → {global_id → mask}}
    id2label = {}          # global_id (int) → label string
    id_offset = 0
    for label, session_masks in zip(labels, all_sessions):
        # Find max local ID used in this session (0 if no detections).
        max_local = max(
            (max(local_ids) for local_ids in session_masks.values() if local_ids),
            default=-1,
        )
        remapped = {}
        for frame_idx, frame_masks in session_masks.items():
            remapped[frame_idx] = {}
            for local_id, mask in frame_masks.items():
                global_id = id_offset + local_id + 1
                remapped[frame_idx][global_id] = mask
                id2label[global_id] = label   # same label for all instances
        global_sessions.append(remapped)
        id_offset += max_local + 1   # reserve IDs used by this session

    # ── 5. Merge sessions per frame and write outputs ─────────────────────
    # Determine full frame index range.
    if frame_stems is not None:
        all_frame_indices = list(range(len(frame_stems)))
    else:
        all_frame_indices = sorted({
            fi for gs in global_sessions for fi in gs
        })

    print(f"  Merging {len(labels)} session(s) across {len(all_frame_indices)} frame(s) …")

    total_written = 0
    for frame_idx in all_frame_indices:
        stem = (frame_stems[frame_idx]
                if frame_stems and frame_idx < len(frame_stems)
                else f"frame_{frame_idx:05d}")

        h = orig_h or 1
        w = orig_w or 1
        id_img = np.zeros((h, w), dtype=np.uint16)

        # Paint sessions in priority order (first = highest frequency).
        # First-session-wins: only paint unclaimed pixels (id_img == 0).
        for gs in global_sessions:
            if frame_idx not in gs:
                continue
            for global_id, mask in gs[frame_idx].items():
                if mask.shape != (h, w):
                    mask = mask[-h:, -w:]   # crop if shape mismatch
                if non_overlap:
                    id_img[mask & (id_img == 0)] = global_id
                else:
                    id_img[mask & (id_img == 0)] = global_id

        n_objects = len(np.unique(id_img)) - 1   # exclude background 0
        if total_written < 5 or n_objects > 0:
            print(f"    frame {frame_idx:5d}: {n_objects} object(s)")

        Image.fromarray(id_img).save(str(gray_dir / f"{stem}.png"))
        Image.fromarray(_id_image_to_color(id_img)).save(
            str(color_dir / f"{stem}_mask.png")
        )
        total_written += 1

    # ── 6. Write id2label.json ────────────────────────────────────────────
    import json
    id2label_path = Path(output_dir) / "id2label.json"
    with id2label_path.open("w") as f:
        # Keys must be strings in JSON; sort numerically for readability.
        json.dump({str(k): v for k, v in sorted(id2label.items())}, f, indent=2)
    print(f"  id2label.json → {id2label_path}")

    print(f"  Done → {total_written} frame(s) written")
    print(f"         gray_mask : {gray_dir}")
    print(f"         color_mask: {color_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "SAM3 scene auto-segmenter.\n"
            "\n"
            "Stage 1: Florence-2-base discovers object vocabulary from sampled frames.\n"
            "Stage 2: SAM3 video predictor segments all frames with consistent IDs.\n"
            "\n"
            "Skip Stage 1 by providing --prompt explicitly."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input (mutually exclusive) ─────────────────────────────────────────
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "--input_dir",
        help="Directory of JPEG/PNG frames (sorted numerically or lexicographically).",
    )
    inp.add_argument(
        "--input_video",
        help="Path to an MP4/AVI/MOV video file.",
    )

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", required=True,
                   help="Directory where mask PNGs and info TXTs are written.")

    # ── Vocabulary discovery (Stage 1) ────────────────────────────────────
    p.add_argument(
        "--prompt",
        default=None,
        help=(
            "Comma-separated text prompt for SAM3, e.g. 'person, car, dog'.\n"
            "If omitted, Florence-2 auto-discovers the vocabulary (Stage 1)."
        ),
    )
    p.add_argument(
        "--vocab_stride",
        type=int,
        default=None,
        metavar="N",
        help=(
            "(Stage 1) Sample every N-th frame for vocabulary discovery. "
            "Defaults to 5 for image directories and 30 for video files."
        ),
    )
    p.add_argument(
        "--vocab_device",
        default=None,
        metavar="DEVICE",
        help=(
            "(Stage 1) Device for Florence-2: 'cuda', 'cpu', or None=auto "
            "(default: auto – uses CUDA if available)."
        ),
    )
    p.add_argument(
        "--min_label_frames",
        type=int,
        default=1,
        metavar="N",
        help=(
            "(Stage 1) Minimum number of sampled frames a label must appear in "
            "to be kept (default: 1).  Raise to filter transient detections."
        ),
    )
    p.add_argument(
        "--min_label_freq",
        type=float,
        default=0.30,
        metavar="F",
        help=(
            "(Stage 1) Minimum fraction of sampled frames a label must appear in "
            "to be kept (default: 0.30 = 30%%).  Labels seen in fewer frames are "
            "treated as transient noise and discarded."
        ),
    )
    p.add_argument(
        "--max_labels",
        type=int,
        default=4,
        metavar="N",
        help=(
            "(Stage 1) Maximum number of categories passed to SAM3 (default: 4). "
            "SAM3 encodes the entire prompt as one CLIP phrase; too many labels "
            "dilute the embedding and reduce detection accuracy."
        ),
    )

    # ── SAM3 options (Stage 2) ─────────────────────────────────────────────
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="SAM3 detector confidence threshold 0-1 (default: 0.3).",
    )
    p.add_argument(
        "--anchor_frame",
        type=int,
        default=0,
        help="Frame index where SAM3 detector fires first (default: 0).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a local SAM3 .pt checkpoint.  "
            "Default: auto-detect ckpt/sam3.pt or ckpt/sam3.1_multiplex.pt "
            "beside this script, then fall back to HuggingFace download."
        ),
    )
    p.add_argument(
        "--non_overlap",
        action="store_true",
        help="Each pixel belongs to at most one mask.",
    )
    p.add_argument(
        "--version",
        choices=["sam3", "sam3.1"],
        default="sam3",
        help="SAM3 model version (default: sam3 — uses ckpt/sam3.pt).",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    input_path = args.input_video or args.input_dir

    # ── Stage 1: Vocabulary discovery ────────────────────────────────────
    if args.prompt:
        prompt = args.prompt
        print(f"[Stage 1/2] Skipped – using explicit prompt: {prompt!r}")
    else:
        is_video = args.input_video is not None
        vocab_stride = args.vocab_stride or (30 if is_video else 5)
        print("[Stage 1/2] Vocabulary discovery with Florence-2-base …")
        labels, raw_label_counts = discover_vocabulary(
            input_path=input_path,
            every_n=vocab_stride,
            device=args.vocab_device,
            min_label_frames=args.min_label_frames,
            min_label_freq=args.min_label_freq,
            top_n=args.max_labels,
        )

        # ── Save Florence-2 raw output report ─────────────────────────────
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(args.output_dir) / "florence2_labels.txt"
        with report_path.open("w") as f:
            f.write("# Florence-2 raw label discovery report\n")
            f.write(f"# Input: {input_path}\n")
            f.write(f"# Sampled every {vocab_stride} frame(s)\n")
            f.write("#\n")
            f.write("# label\tframes_seen\n")
            for lbl, cnt in sorted(raw_label_counts.items(),
                                   key=lambda kv: kv[1], reverse=True):
                f.write(f"{lbl}\t{cnt}\n")
            f.write(f"\n# Final prompt passed to SAM3 (after filtering + dedup + top-N):\n")
        print(f"  Florence-2 raw labels saved → {report_path}")

        if not labels:
            raise RuntimeError(
                "Florence-2 detected no objects in the sampled frames.\n"
                "Options:\n"
                "  • Lower --vocab_stride to sample more frames\n"
                "  • Lower --threshold\n"
                "  • Provide --prompt manually (e.g. --prompt 'person, car')"
            )
        clean = _deduplicate_labels(labels)
        dropped_sub = labels - clean
        if dropped_sub:
            print(f"  Removed {len(dropped_sub)} sub-label(s): {sorted(dropped_sub)}")
        prompt = labels_to_sam3_prompt(labels, label_counts=raw_label_counts)
        print(f"  → {len(clean)} label(s) after dedup: {prompt!r}")

        # Append the final prompt to the report
        with report_path.open("a") as f:
            f.write(f"# {prompt}\n")

    # ── Stage 2: SAM3 segmentation ────────────────────────────────────────
    print("[Stage 2/2] SAM3 video segmentation …")
    segment_with_sam3(
        input_path=input_path,
        output_dir=args.output_dir,
        prompt=prompt,
        threshold=args.threshold,
        anchor_frame=args.anchor_frame,
        checkpoint_path=args.checkpoint,
        non_overlap=args.non_overlap,
        version=args.version,
    )


if __name__ == "__main__":
    main()