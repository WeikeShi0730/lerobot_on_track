"""Add the YOLO target box step to a pi05 preprocessor that was trained on a `bake_boxes.py` dataset.

    python examples/yolo_target/add_yolo_step.py \
        --pretrained-model outputs/pi05_training/<run>/checkpoints/last/pretrained_model \
        --yolo-model outputs/yolo_target/runs/final2/weights/best.pt

The step goes right after the batch step, the same place as with `--policy.yolo_model`.
"""

import argparse
import json
from pathlib import Path

from lerobot.policies.pi05.processor_pi05 import YoloTargetBoxProcessorStep

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-model", type=Path, required=True)
parser.add_argument("--yolo-model", required=True)
parser.add_argument("--camera-key", default="observation.images.base", help="dataset key, before --rename_map")
args = parser.parse_args()

path = args.pretrained_model / "policy_preprocessor.json"
pipeline = json.loads(path.read_text())
names = [s["registry_name"] for s in pipeline["steps"]]
if "pi05_yolo_target_box_processor_step" in names:
    raise SystemExit(f"{path} already has the YOLO step")

rename_map = pipeline["steps"][names.index("rename_observations_processor")]["config"]["rename_map"]
step = YoloTargetBoxProcessorStep(model_path=args.yolo_model, camera_key=rename_map.get(args.camera_key, args.camera_key))
pipeline["steps"].insert(
    names.index("to_batch_processor") + 1,
    {"registry_name": "pi05_yolo_target_box_processor_step", "config": step.get_config()},
)
path.write_text(json.dumps(pipeline, indent=2))
print(f"Added YOLO step ({step.camera_key}) to {path}")
