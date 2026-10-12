"""Draw the pi05 YOLO target box into the base camera videos of a dataset copy.

Training on the copy needs no YOLO on the GPU. The boxes come from the same
`YoloTargetBoxProcessorStep` that runs at rollout, so training and rollout match.

    python examples/yolo_target/bake_boxes.py --repo-id vicshi97/sock-garbage-merged_2.0 \
        --yolo-model outputs/yolo_target/runs/final2/weights/best.pt \
        --out outputs/datasets/sock-garbage-merged_2.0_yolo

Train with `--dataset.root=<out>` and without `--policy.yolo_model`. For rollout, add the
YOLO step to the trained preprocessor with `add_yolo_step.py`.
"""

import argparse
import json
import shutil
from pathlib import Path

import av
import numpy as np
import torch
from av.codec.hwaccel import HWAccel

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi05.processor_pi05 import YoloTargetBoxProcessorStep

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--yolo-model", required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--camera-key", default="observation.images.base")
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--vcodec", default="hevc_nvenc", help="hevc_nvenc: NVIDIA GPU, libsvtav1: CPU")
args = parser.parse_args()

meta = LeRobotDataset(args.repo_id).meta  # downloads the videos if needed
cam = args.camera_key
shutil.copytree(meta.root, args.out, ignore=lambda d, names: [cam] if Path(d).name == "videos" else [])
device = "cuda" if torch.cuda.is_available() else "cpu"
step = YoloTargetBoxProcessorStep(model_path=args.yolo_model, camera_key=cam)
info = meta.info["features"][cam]["info"]
# NVENC needs no B-frames for a GOP of 2.
encoder = RGBEncoderConfig(
    vcodec=args.vcodec,
    g=info["video.g"],
    crf=info["video.crf"],
    extra_options={"bf": 0} if "nvenc" in args.vcodec else {},
)


def box_and_encode(frames, tasks, stream, dst):
    images = torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2).float() / 255
    transition = {TransitionKey.OBSERVATION: {cam: images}, TransitionKey.COMPLEMENTARY_DATA: {"task": tasks}}
    images = step(transition)[TransitionKey.OBSERVATION][cam]
    for image in (images * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy():
        dst.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))


for video in sorted((meta.root / "videos" / cam).glob("*/*.mp4")):
    chunk, file = int(video.parent.name.split("-")[1]), int(video.stem.split("-")[1])
    spans = [
        (ep[f"videos/{cam}/from_timestamp"], ep[f"videos/{cam}/to_timestamp"], ep["tasks"][0])
        for ep in meta.episodes
        if ep[f"videos/{cam}/chunk_index"] == chunk and ep[f"videos/{cam}/file_index"] == file
    ]
    out_path = args.out / video.relative_to(meta.root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hwaccel = HWAccel(device_type="cuda") if device == "cuda" else None  # NVDEC decoding
    with (
        av.open(str(video), hwaccel=hwaccel) as src,
        av.open(str(out_path), "w", options={"movflags": "faststart"}) as dst,
    ):
        in_stream = src.streams.video[0]
        options = encoder.get_codec_options(as_strings=True)
        stream = dst.add_stream(encoder.vcodec, in_stream.base_rate, options=options)
        stream.pix_fmt, stream.width, stream.height = encoder.pix_fmt, in_stream.width, in_stream.height
        frames, tasks, n = [], [], 0
        for frame in src.decode(in_stream):
            # Frames outside all episodes get an empty task, so no box.
            tasks.append(next((t for a, b, t in spans if a - 1e-4 <= frame.time < b - 1e-4), ""))
            frames.append(frame.to_ndarray(format="rgb24"))
            n += 1
            if len(frames) == args.batch:
                box_and_encode(frames, tasks, stream, dst)
                frames, tasks = [], []
        if frames:
            box_and_encode(frames, tasks, stream, dst)
        dst.mux(stream.encode())
    print(f"{out_path}: {n} frames")

info_path = args.out / "meta" / "info.json"
new_info = json.loads(info_path.read_text())
new_info["features"][cam]["info"].update({"video.codec": args.vcodec.removesuffix("_nvenc"), "video.preset": None})
info_path.write_text(json.dumps(new_info, indent=4))
