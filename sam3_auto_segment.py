"""
SAM3 Automatic Segmentation Script
====================================
Two operating modes:

  --mode image  (default)
      Treat each file as an INDEPENDENT image.
      IDs are NOT consistent across images (a car in img_001.jpg may get a
      different ID in img_002.jpg). Use for unrelated photos.

  --mode video  (USE THIS FOR FRAME SEQUENCES)
      Treat the directory as an ordered VIDEO SEQUENCE.
      The SAM 3.1 video predictor + memory tracker is used.
      The same physical object keeps the SAME ID across all frames.

Usage
-----
  # Unrelated images:
  python sam3_auto_segment.py \\
      --mode image --input_dir /path/to/photos --output_dir /out

  # Video sequence (consistent IDs):
  python sam3_auto_segment.py \\
      --mode video --input_dir /path/to/frames --output_dir /out

  # Or from a video file:
  python sam3_auto_segment.py \\
      --mode video --input_video clip.mp4 --output_dir /out

Requirements
------------
  pip install -e ".[notebooks]"
  hf auth login   # to auto-download the SAM 3.1 checkpoint

Output (per frame/image)
------------------------
  <stem>_mask.png  – 16-bit grayscale, pixel = object ID (0 = background)
  <stem>_info.txt  – TSV: id | score | x0 | y0 | x1 | y1

Frame ordering (video mode, directory input)
---------------------------------------------
  Frames are sorted NUMERICALLY if all stems are integers (00000.jpg, 00001.jpg…)
  and LEXICOGRAPHICALLY otherwise. This matches SAM3's internal io_utils behaviour.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint resolver
# ─────────────────────────────────────────────────────────────────────────────

# Canonical local paths to look for before falling back to HuggingFace download.
# Keys match the --version / build_sam3_predictor `version` argument.
_LOCAL_CKPT_CANDIDATES = {
    "sam3":   ["ckpt/sam3.pt",           "checkpoints/sam3.pt"],
    "sam3.1": ["ckpt/sam3.1_multiplex.pt", "checkpoints/sam3.1_multiplex.pt"],
}

# Script's own directory so relative paths work wherever cwd is.
_SCRIPT_DIR = Path(__file__).parent


def _resolve_checkpoint(checkpoint_path: str | None, version: str) -> tuple[str | None, bool]:
    """
    Return (resolved_path, load_from_HF).

    Priority:
      1. Explicitly supplied --checkpoint  → use as-is, never download
      2. Known local candidate paths       → use first one that exists
      3. Nothing found                     → return (None, True) to trigger HF download
    """
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
    print("    (Run `hf auth login` first if you have not authenticated yet.)")
    return None, True


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_id_image(obj_ids, binary_masks, height, width, non_overlap: bool = False):
    """
    Rasterise (obj_id, mask) pairs into a uint16 ID image.

    obj_ids      : array-like [N] int   — ID values to paint (any positive int)
    binary_masks : array-like [N, H, W] bool
    non_overlap  : if True, once a pixel is claimed it cannot be overwritten
                   (first in list = highest priority)
    """
    id_image = np.zeros((height, width), dtype=np.uint16)
    if len(obj_ids) == 0:
        return id_image

    masks_np = np.asarray(binary_masks, dtype=bool)   # [N, H, W]
    ids_np   = np.asarray(obj_ids,      dtype=np.int32)

    if non_overlap:
        for obj_id, mask in zip(ids_np, masks_np):
            free = id_image == 0
            id_image[free & mask] = int(obj_id)
    else:
        # Write in reverse order so first entry (highest priority) wins on overlap
        for obj_id, mask in zip(ids_np[::-1], masks_np[::-1]):
            id_image[mask] = int(obj_id)

    return id_image


def save_info_txt(path, obj_ids, scores, boxes_xywh, orig_h, orig_w):
    """TSV: id | score | x0 | y0 | x1 | y1  (absolute pixel coordinates)."""
    lines = ["id\tscore\tx0\ty0\tx1\ty1"]
    for obj_id, score, box in zip(obj_ids, scores, boxes_xywh):
        x, y, w, h = (float(v) for v in box)
        lines.append(
            f"{int(obj_id)}\t{float(score):.4f}"
            f"\t{x * orig_w:.1f}\t{y * orig_h:.1f}"
            f"\t{(x + w) * orig_w:.1f}\t{(y + h) * orig_h:.1f}"
        )
    Path(path).write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# MODE A – independent images  (no cross-frame tracking)
# ─────────────────────────────────────────────────────────────────────────────

def segment_images_independent(
    input_dir: str,
    output_dir: str,
    prompt: str = "everything",
    threshold: float = 0.3,
    checkpoint_path: str = None,
    non_overlap: bool = False,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
):
    """
    Process each image independently.  IDs are NOT stable across images.

    Uses build_sam3_image_model + Sam3Processor (no memory, no tracking).
    Assign IDs 1..N by descending confidence score.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[1/4] Building SAM3 image model …")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    resolved_ckpt, load_from_hf = _resolve_checkpoint(checkpoint_path, version="sam3")
    model = build_sam3_image_model(
        checkpoint_path=resolved_ckpt,
        load_from_HF=load_from_hf,
        eval_mode=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    processor = Sam3Processor(
        model=model,
        confidence_threshold=threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("    Model ready.")

    print("[2/4] Scanning input directory …")
    image_paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in extensions
    )
    if not image_paths:
        print(f"    No images found in {input_dir!r}.")
        return
    print(f"    Found {len(image_paths)} image(s).")

    print(f"[3/4] Segmenting (prompt='{prompt}') …")
    for img_path in image_paths:
        print(f"  {img_path.name} …", end=" ", flush=True)

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=prompt, state=state)

        masks  = state.get("masks")   # [N, 1, H, W] bool tensor
        scores = state.get("scores")  # [N] tensor
        boxes  = state.get("boxes")   # [N, 4] pixel XYXY tensor
        n = len(scores) if scores is not None else 0
        print(f"{n} object(s).")

        order    = np.empty(0, dtype=np.int64)
        obj_ids  = np.empty(0, dtype=np.int32)
        id_image = np.zeros((H, W), dtype=np.uint16)

        if n > 0 and masks is not None and scores is not None:
            # Sort descending by score → ID 1 = highest confidence
            order        = torch.argsort(scores, descending=True).cpu().numpy()
            obj_ids      = np.arange(1, n + 1, dtype=np.int32)
            sorted_masks = (masks[order].squeeze(1).cpu().numpy() > 0.5)  # [N,H,W]
            id_image     = build_id_image(obj_ids, sorted_masks, H, W, non_overlap)

        Image.fromarray(id_image, mode="I;16").save(
            str(Path(output_dir) / f"{img_path.stem}_mask.png")
        )
        if boxes is not None and len(order) > 0:
            boxes_np = boxes.cpu().numpy()
            xywh = np.stack([
                boxes_np[:, 0] / W,
                boxes_np[:, 1] / H,
                (boxes_np[:, 2] - boxes_np[:, 0]) / W,
                (boxes_np[:, 3] - boxes_np[:, 1]) / H,
            ], axis=1)[order]
            save_info_txt(
                Path(output_dir) / f"{img_path.stem}_info.txt",
                obj_ids, scores[order].cpu().numpy(), xywh, H, W,  # type: ignore[index]
            )

    print("[4/4] Done →", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# MODE B – video / image sequence  (cross-frame consistent IDs via tracking)
# ─────────────────────────────────────────────────────────────────────────────

def segment_video_sequence(
    input_path: str,
    output_dir: str,
    prompt: str = "everything",
    threshold: float = 0.3,
    anchor_frame: int = 0,
    checkpoint_path: str = None,
    non_overlap: bool = False,
    version: str = "sam3.1",
):
    """
    Segment a frame sequence with CONSISTENT cross-frame object IDs.

    How it works
    ------------
    1. build_sam3_predictor() builds the detector + memory tracker.
    2. init_state() (via start_session) loads the entire frame sequence.
    3. add_prompt() fires the open-vocabulary detector on `anchor_frame`.
       No spatial prompts → fully automatic discovery.
    4. propagate_in_video() runs the tracker forward AND backward from
       anchor_frame, yielding one output dict per frame.
    5. outputs["out_obj_ids"] are STABLE integers from the tracker.
       ID 3 in frame 0 is guaranteed to be the same object as ID 3 in frame 99.

    Input path
    ----------
    - A directory of JPEG/PNG frames sorted numerically (00000.jpg, 00001.jpg…)
      or lexicographically if stems are not all integers.
    - Or an MP4/AVI/MOV video file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Build predictor ────────────────────────────────────────────────
    print(f"[1/4] Building SAM3 {version} video predictor …")
    from sam3.model_builder import build_sam3_predictor
    resolved_ckpt, load_from_hf = _resolve_checkpoint(checkpoint_path, version)
    predictor = build_sam3_predictor(
        checkpoint_path=resolved_ckpt,
        version=version,
        use_fa3=False,           # set True if flash-attn-3 is installed
        async_loading_frames=True,
    )
    print("    Predictor ready.")

    # ── 2. Start session (loads all frames) ───────────────────────────────
    print("[2/4] Loading frames …")
    response   = predictor.handle_request({
        "type": "start_session",
        "resource_path": input_path,
    })
    session_id = response["session_id"]

    # Build stem lookup for output file naming (mirrors io_utils sort logic)
    input_obj = Path(input_path)
    if input_obj.is_dir():
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        all_frames = [p for p in input_obj.iterdir()
                      if p.suffix.lower() in IMAGE_EXTS]
        try:
            all_frames.sort(key=lambda p: int(p.stem))
        except ValueError:
            all_frames.sort()
        frame_stems = [p.stem for p in all_frames]
    else:
        frame_stems = None   # video file – use index-based names

    # ── Determine frame dimensions up-front so zero-detection frames still
    #    produce correctly-sized masks (not 1×1).
    orig_h = orig_w = None
    if input_obj.is_dir() and all_frames:
        with Image.open(all_frames[0]) as _img:
            orig_w, orig_h = _img.size   # PIL gives (W, H)
    elif not input_obj.is_dir():
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(str(input_obj))
            orig_w = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            _cap.release()
        except Exception:
            pass   # will fall back to mask-derived dims later

    # ── 3. Add automatic text prompt on anchor frame ───────────────────────
    print(f"[3/4] Detecting '{prompt}' on frame {anchor_frame} …")
    predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": anchor_frame,
        "text": prompt,
        "output_prob_thresh": threshold,
    })

    # ── 4. Propagate through all frames and save ──────────────────────────
    print("[4/4] Propagating to all frames …")

    for result in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "both",   # forward + backward from anchor_frame
        "output_prob_thresh": threshold,
    }):
        frame_idx    = result["frame_index"]
        outputs      = result["outputs"]

        # These are the STABLE cross-frame IDs maintained by the tracker.
        obj_ids      = outputs["out_obj_ids"]        # np.ndarray [N] int
        binary_masks = outputs["out_binary_masks"]   # np.ndarray [N, H, W] bool
        probs        = outputs["out_probs"]          # np.ndarray [N] float
        boxes_xywh   = outputs["out_boxes_xywh"]     # np.ndarray [N, 4] norm. XYWH

        if orig_h is None and len(binary_masks) > 0:
            orig_h, orig_w = binary_masks.shape[-2], binary_masks.shape[-1]

        stem = (frame_stems[frame_idx]
                if frame_stems and frame_idx < len(frame_stems)
                else f"frame_{frame_idx:05d}")

        n = len(obj_ids)
        print(f"  frame {frame_idx:5d}: {n} object(s) "
              f"ids={list(obj_ids[:6])}{'…' if n > 6 else ''}")

        if n == 0 or orig_h is None:
            h = orig_h or 1
            w = orig_w or 1
            id_image = np.zeros((h, w), dtype=np.uint16)
        else:
            # Use the tracker's obj_ids DIRECTLY as pixel values.
            # They are stable integers — no remapping needed.
            id_image = build_id_image(
                obj_ids, binary_masks, orig_h, orig_w, non_overlap=non_overlap
            )

        Image.fromarray(id_image, mode="I;16").save(
            str(Path(output_dir) / f"{stem}_mask.png")
        )
        if n > 0 and orig_h is not None:
            save_info_txt(
                Path(output_dir) / f"{stem}_info.txt",
                obj_ids, probs, boxes_xywh, orig_h, orig_w,
            )

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    print("Done →", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "SAM3 automatic segmentation.\n"
            "  --mode image  independent images (IDs NOT stable across files)\n"
            "  --mode video  frame sequence    (IDs ARE stable across frames)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["image", "video"], default="image",
                   help="'image' or 'video' (default: image)")

    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input_dir",
                     help="Directory of images (image mode) or JPEG frames (video mode).")
    inp.add_argument("--input_video",
                     help="(video mode only) path to an MP4/AVI/MOV file.")

    p.add_argument("--output_dir",   required=True)
    p.add_argument("--prompt",       default="objects",
                   help="Broad text prompt: 'everything', 'object', etc. (default: everything)")
    p.add_argument("--threshold",    type=float, default=0.3,
                   help="Confidence threshold 0-1 (default 0.3)")
    p.add_argument("--checkpoint",   default=None,
                   help="Path to a local .pt checkpoint (default: auto-detect "
                        "ckpt/sam3.pt or ckpt/sam3.1_multiplex.pt beside the script, "
                        "then fall back to HuggingFace download).")
    p.add_argument("--non_overlap",  action="store_true",
                   help="Each pixel belongs to at most one mask")
    p.add_argument("--anchor_frame", type=int, default=0,
                   help="(video mode) Frame where the detector fires first (default 0)")
    p.add_argument("--version",      choices=["sam3", "sam3.1"], default="sam3",
                   help="(video mode) 'sam3' uses sam3.pt, 'sam3.1' uses "
                        "sam3.1_multiplex.pt (default: sam3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "image":
        if args.input_dir is None:
            raise ValueError("--input_dir required for image mode.")
        segment_images_independent(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            prompt=args.prompt,
            threshold=args.threshold,
            checkpoint_path=args.checkpoint,
            non_overlap=args.non_overlap,
        )
    else:  # video
        input_path = args.input_video or args.input_dir
        if input_path is None:
            raise ValueError("Provide --input_dir or --input_video for video mode.")
        segment_video_sequence(
            input_path=input_path,
            output_dir=args.output_dir,
            prompt=args.prompt,
            threshold=args.threshold,
            anchor_frame=args.anchor_frame,
            checkpoint_path=args.checkpoint,
            non_overlap=args.non_overlap,
            version=args.version,
        )
