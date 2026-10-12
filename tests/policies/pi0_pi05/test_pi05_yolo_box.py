from types import SimpleNamespace

import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi05.processor_pi05 import YoloTargetBoxProcessorStep

CAM = "observation.images.base"


class FakeYolo:
    """Each image has a garbage box (higher conf) and a sock box."""

    names = {0: "sock", 1: "garbage"}

    def predict(self, images, **kwargs):
        boxes = SimpleNamespace(
            cls=torch.tensor([1.0, 0.0]),
            conf=torch.tensor([0.9, 0.6]),
            xyxy=torch.tensor([[100.0, 100.0, 200.0, 200.0], [10.0, 20.0, 50.0, 60.0]]),
        )
        return [SimpleNamespace(boxes=boxes) for _ in images]


def run(tasks):
    step = YoloTargetBoxProcessorStep(model_path="fake.pt")
    step._model = FakeYolo()
    images = torch.zeros(len(tasks), 3, 480, 640)
    out = step(
        {
            TransitionKey.OBSERVATION: {CAM: images},
            TransitionKey.COMPLEMENTARY_DATA: {"task": tasks},
        }
    )
    assert images.abs().sum() == 0, "input image must not change in place"
    return out[TransitionKey.OBSERVATION][CAM]


def test_box_on_prompt_target_only():
    out = run(["pick up the sock and put it in the container", "pick up the towel"])
    sock = out[0]
    # Green outline on the sock box (10, 20, 50, 60), nothing on the garbage box.
    assert sock[1, 20, 30] == 1 and sock[0, 20, 30] == 0
    assert sock[1, 40, 30] == 0, "box interior must stay unchanged"
    assert sock[:, 100:200, 100:200].sum() == 0
    # No class name in the prompt: image unchanged.
    assert out[1].sum() == 0
