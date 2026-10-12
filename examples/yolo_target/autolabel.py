"""Draft-label base camera frames with Grounding DINO, for review and custom `yolo11n` training.

Runs in its own env, without lerobot:

    uv tool run --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
        --with "torch==2.11.0+cu128" --with torchvision --with transformers --with opencv-python --with huggingface_hub \
        python examples/yolo_target/autolabel.py --repo-id vicshi97/sock-garbage-merged_2.0 --out outputs/yolo_target

Without an NVIDIA GPU, remove the --index options and the "+cu128" (CPU, much slower).

Output is a YOLO dataset (images/, labels/, data.yaml). The drafts have errors: review and fix
them (e.g. X-AnyLabeling or Roboflow, YOLO format), then train:

    yolo detect train data=outputs/yolo_target/data.yaml model=yolo11n.pt epochs=100 imgsz=640

Class names must be words that occur in the task prompts ("sock", "garbage"):
the pi05 YOLO step finds the target by matching class names against the prompt.
"""

import argparse
from pathlib import Path

import cv2
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Class id -> (name in the prompt, Grounding DINO query). One query per call: a joint
# query mixes up the classes much more often.
CLASSES = {0: ("sock", "a sock."), 1: ("garbage", "a crumpled ball of paper.")}
MAX_BOX_AREA = 0.4  # drop boxes larger than this fraction of the frame (floor, bins)


def detect(model, proc, frame, conf):
    """Return [(class_id, x1, y1, x2, y2)] for one BGR frame."""
    h, w = frame.shape[:2]
    image = Image.fromarray(frame[:, :, ::-1])
    boxes, scores, classes = [], [], []
    for class_id, (_, query) in CLASSES.items():
        inputs = proc(images=image, text=query, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=conf, text_threshold=conf, target_sizes=[(h, w)]
        )[0]
        for box, score in zip(result["boxes"].cpu(), result["scores"].cpu(), strict=True):
            if (box[2] - box[0]) * (box[3] - box[1]) < MAX_BOX_AREA * h * w:
                boxes.append(box)
                scores.append(score)
                classes.append(class_id)
    if not boxes:
        return []
    keep = nms(torch.stack(boxes), torch.stack(scores), iou_threshold=0.5)
    return [(classes[i], *boxes[i].tolist()) for i in keep.tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--camera-key", default="observation.images.base")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=180, help="keep one frame every N frames")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    args = parser.parse_args()

    root = Path(
        snapshot_download(args.repo_id, repo_type="dataset", allow_patterns=[f"videos/{args.camera_key}/*"])
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    n = 0
    for video in sorted(root.glob(f"videos/{args.camera_key}/**/*.mp4")):
        cap = cv2.VideoCapture(str(video))
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.every == 0:
                # ponytail: frame-level split leaks near-duplicate scenes into val; split by episode if val mAP looks too good
                split = "val" if n % 5 == 0 else "train"
                name = f"{video.parent.name}_{video.stem}_{frame_idx:06d}"
                (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
                (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.out / "images" / split / f"{name}.jpg"), frame)

                h, w = frame.shape[:2]
                lines = [
                    f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
                    for c, x1, y1, x2, y2 in detect(model, proc, frame, args.conf)
                ]
                (args.out / "labels" / split / f"{name}.txt").write_text("".join(lines))
                n += 1
                if n % 50 == 0:
                    print(f"{n} frames")
            frame_idx += 1
        cap.release()

    names = "\n".join(f"  {i}: {name}" for i, (name, _) in CLASSES.items())
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    print(f"Wrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()
