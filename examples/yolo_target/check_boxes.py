"""Save random base camera frames with the pi05 YOLO target box, to check the box before training.

Run in the training env (`uv sync --locked --extra pi --extra training --extra yolo`):

    python examples/yolo_target/check_boxes.py --repo-id vicshi97/sock-garbage-merged_2.0 \
        --yolo-model outputs/yolo_target/runs/final2/weights/best.pt --out outputs/yolo_target/check.jpg

For a `bake_boxes.py` dataset, pass `--root <out>` and no `--yolo-model`: the boxes are in the video.

The grid is row-major; the task of each frame is printed in the same order.
"""

import argparse

import torch
from torchvision.utils import save_image

from lerobot.datasets import LeRobotDataset
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi05.processor_pi05 import YoloTargetBoxProcessorStep

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--yolo-model")
parser.add_argument("--root")
parser.add_argument("--camera-key", default="observation.images.base")
parser.add_argument("--n", type=int, default=16)
parser.add_argument("--out", default="check.jpg")
args = parser.parse_args()

dataset = LeRobotDataset(args.repo_id, root=args.root)
indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[: args.n]
items = [dataset[int(i)] for i in indices]
tasks = [item["task"] for item in items]

images = torch.stack([item[args.camera_key] for item in items])
if args.yolo_model:
    step = YoloTargetBoxProcessorStep(model_path=args.yolo_model, camera_key=args.camera_key)
    transition = {
        TransitionKey.OBSERVATION: {args.camera_key: images},
        TransitionKey.COMPLEMENTARY_DATA: {"task": tasks},
    }
    images = step(transition)[TransitionKey.OBSERVATION][args.camera_key]
save_image(images, args.out, nrow=4)
for i, task in enumerate(tasks):
    print(i, task)
print(f"Saved {args.out}")
