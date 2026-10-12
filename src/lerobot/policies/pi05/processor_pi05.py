#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_pi05 import PI05Config


def _draw_box(image: torch.Tensor, xyxy: list[float], thickness: int = 6) -> None:
    """Draw a green box outline in place on a [C, H, W] image in [0, 1].

    6 px at 640 wide is still ~2 px after the resize to 224 inside the model.
    """
    _, h, w = image.shape
    x1, y1 = max(int(xyxy[0]), 0), max(int(xyxy[1]), 0)
    x2, y2 = min(int(xyxy[2]), w), min(int(xyxy[3]), h)
    color = torch.tensor([0.0, 1.0, 0.0], dtype=image.dtype, device=image.device).view(3, 1, 1)
    image[:, y1 : y1 + thickness, x1:x2] = color
    image[:, max(y2 - thickness, y1) : y2, x1:x2] = color
    image[:, y1:y2, x1 : x1 + thickness] = color
    image[:, y1:y2, max(x2 - thickness, x1) : x2] = color


@ProcessorStepRegistry.register(name="pi05_yolo_target_box_processor_step")
@dataclass
class YoloTargetBoxProcessorStep(ProcessorStep):
    """Draw a box around the object that the task prompt names, so the policy knows which object to grab.

    The target is each YOLO class name that occurs in the task string, e.g. "sock" in
    "pick up the sock and put it in the container". Only the top-scoring match gets a box.
    If YOLO finds no match, the frame stays unchanged.
    """

    model_path: str = ""
    camera_key: str = "observation.images.base"
    conf: float = 0.25
    device: str | None = None
    task_key: str = "task"
    _model: Any = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        tasks = (transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}).get(self.task_key)
        if self.camera_key not in observation:
            raise KeyError(
                f"YOLO camera key '{self.camera_key}' not in observation keys {list(observation)}. "
                "If you use --rename_map, set --policy.yolo_camera_key to the renamed key."
            )
        if tasks is None:
            return transition
        if isinstance(tasks, str):
            tasks = [tasks]

        images = observation[self.camera_key]
        if images.ndim != 4:
            raise ValueError(
                f"Expected [B, C, H, W] images for '{self.camera_key}', got {tuple(images.shape)}"
            )

        if self._model is None:
            require_package("ultralytics", extra="yolo")
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)

        results = self._model.predict(images, conf=self.conf, device=self.device, verbose=False)
        names: dict[int, str] = self._model.names
        images = images.clone()
        for image, task, result in zip(images, tasks, results, strict=True):
            target_ids = [i for i, name in names.items() if name in task.lower()]
            boxes = result.boxes
            match = torch.isin(boxes.cls.long(), torch.tensor(target_ids, device=boxes.cls.device))
            if match.any():
                best = boxes.conf.masked_fill(~match, -1).argmax()
                _draw_box(image, boxes.xyxy[best].tolist())

        transition = transition.copy()
        transition[TransitionKey.OBSERVATION] = {**observation, self.camera_key: images}
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "camera_key": self.camera_key,
            "conf": self.conf,
            "device": self.device,
            "task_key": self.task_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    # MEM section III-D represents proprioception with a linear projection into the
    # backbone instead of discretized prompt tokens, so the state is carried once.
    # Set from `PI05Config.use_proprioceptive_memory`; stock PI0.5 keeps it in the prompt.
    include_state_in_prompt: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        observation = transition.get(TransitionKey.OBSERVATION)
        state = observation.get(OBS_STATE) if observation is not None else None
        if state is None:
            raise ValueError("State is required for PI05")
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is required for PI05")
        tasks = complementary_data.get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        discretized_states = None
        if self.include_state_in_prompt:
            # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
            # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
            prompt_state = state[:, -1] if state.ndim == 3 else state
            state_np = prompt_state.cpu().numpy()
            discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            if discretized_states is None:
                full_prompt = f"Task: {cleaned_text};\nAction: "
            else:
                state_str = " ".join(map(str, discretized_states[i]))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        complementary_data[self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            include_state_in_prompt=not config.use_proprioceptive_memory,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]
    if config.yolo_model:
        yolo_step = YoloTargetBoxProcessorStep(
            model_path=config.yolo_model,
            camera_key=config.yolo_camera_key,
            conf=config.yolo_conf,
            device=config.device,
        )
        input_steps.insert(input_steps.index(steps.add_batch_dim) + 1, yolo_step)

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
