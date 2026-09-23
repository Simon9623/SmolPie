# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

.PHONY: tests

PYTHON_PATH := $(shell which python)

# If uv is installed and a virtual environment exists, use it
UV_CHECK := $(shell command -v uv)
ifneq ($(UV_CHECK),)
	PYTHON_PATH := $(shell .venv/bin/python)
endif

export PATH := $(dir $(PYTHON_PATH)):$(PATH)

DEVICE ?= cpu

build-user:
	docker build -f docker/Dockerfile.user -t lerobot-user .

build-internal:
	docker build -f docker/Dockerfile.internal -t lerobot-internal .

test-end-to-end:
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-train
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-train-resume
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-eval

test-smolvla-ete-train:
	lerobot-train \
		--policy.type=smolvla \
		--policy.n_action_steps=20 \
		--policy.chunk_size=20 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=4 \
		--save_freq=2 \
		--save_checkpoint=true \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/smolvla/

test-smolvla-ete-train-resume:
	lerobot-train \
		--config_path=tests/outputs/smolvla/checkpoints/000002/pretrained_model/train_config.json \
		--resume=true

# Inference smoke test: loads the checkpoint saved above and runs a real
# select_action() forward pass on a dataset frame. This is the training +
# inference-testing loop documented in AGENT_GUIDE.md — no sim env involved.
test-smolvla-ete-eval:
	uv run python utils/smolvla_inference_smoke_test.py \
		--policy_path=tests/outputs/smolvla/checkpoints/000004/pretrained_model \
		--dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
		--device=$(DEVICE)

# E2E annotation pipeline smoke test against a tiny in-memory fixture
# dataset. Opt-in (not part of `make test-end-to-end`) and uses a stub VLM
# backend, so it does not require a real model checkpoint or GPU.
annotation-e2e:
	uv run python -m tests.annotations.run_e2e_smoke

# Docstring & doctest checks. See docs/source/writing_docstrings.mdx for the standard these enforce.

# Run the examples in the docstrings listed in utils/documentation_tests.txt. Hardware and GPU examples are
# skipped by content (see src/lerobot/utils/doctest_utils.py); CI sets both flags.
doctest:
	@files=$$(grep -v '^\s*#' utils/documentation_tests.txt | grep -v '^\s*$$'); \
	if [ -z "$$files" ]; then \
		echo "utils/documentation_tests.txt lists no files; nothing to run."; \
	else \
		SKIP_HARDWARE_DOCTEST=1 uv run pytest --doctest-modules --no-header -q $$files; \
	fi

check-doctest-list:
	uv run python utils/check_doctest_list.py

fix-doctest-list:
	uv run python utils/check_doctest_list.py --fix_and_overwrite

check-docstrings:
	uv run python utils/check_docstrings.py
	uv run python utils/check_config_docstrings.py

fix-docstrings:
	uv run python utils/check_docstrings.py --fix_and_overwrite
	uv run python utils/check_doctest_list.py --fix_and_overwrite
