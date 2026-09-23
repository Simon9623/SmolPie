#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Loads a trained SmolVLA checkpoint and runs one real select_action() forward pass.

Used by `make test-smolvla-ete-eval` as the inference half of the end-to-end
smolVLA train + inference smoke test (no sim env or lerobot-eval involved).
"""

import argparse

import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--dataset_repo_id", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=args.policy_path)
    dataset = LeRobotDataset(args.dataset_repo_id, episodes=[0])
    batch = dataset[0]
    batch = {k: (v.unsqueeze(0).to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    batch = preprocessor(batch)

    with torch.no_grad():
        action = policy.select_action(batch)

    assert action is not None and action.numel() > 0
    print(f"smolvla inference OK, action shape: {tuple(action.shape)}")


if __name__ == "__main__":
    main()
