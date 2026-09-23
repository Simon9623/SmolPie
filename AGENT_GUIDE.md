# AGENT_GUIDE.md — LeRobot Helper for AI Agents & Users

This file is a practical, copy-paste-friendly companion for any AI agent (Cursor, Claude, ChatGPT, Codex, etc.) helping a user work with LeRobot. It complements [`AGENTS.md`](./AGENTS.md) (dev/contributor context) with **user-facing guidance**: how to start, how to record, and how to train and inference-test **SmolVLA**.

> This is a trimmed fork focused solely on SmolVLA training and real-robot inference testing. Other policies, sim-benchmark evaluation (`lerobot-eval`), and RL training were removed — see `AGENTS.md` for what's in/out of scope.

---

## 1. Start here — ask the user first (MANDATORY)

Before suggesting any command, an agent MUST ask the user at least these questions and wait for answers:

1. **What's your goal?** (e.g. "teach my SO-101 to fold a cloth", "train a policy on an existing HF dataset", "contribute a PR", "understand the codebase")
2. **What hardware do you have?**
   - Robot: none / SO-100 / SO-101 / Koch / LeKiwi / Reachy / other
   - Teleop: leader arm / phone / keyboard / gamepad / none
   - Cameras: how many, resolution, fixed or moving?
3. **What machine will you train on?**
   - GPU model + VRAM (e.g. "laptop 3060 6 GB", "RTX 4090 24 GB", "A100 80 GB", "CPU only")
   - OS: macOS / Linux / Windows
4. **Skill level & time budget?** First time, some ML, experienced? Hours, days, a weekend?
5. **Do you already have a dataset?** Yes (HF repo id?) / no / want to record one
6. **How can I help right now?** (pick one concrete next step)

Only after you have answers, propose a concrete path. If something is ambiguous, ask again rather than guessing. Bias toward **the simplest thing that works** for the user's hardware and goal.

---

## 2. LeRobot in 60 seconds

LeRobot = **datasets + policies + envs + robot control**, unified by a small set of strong abstractions.

- **`LeRobotDataset`** — episode-aware dataset (video or images + actions + state), loadable from the Hub or disk.
- **Policies** — this fork only ships `SmolVLA`, a small vision-language-action model. It inherits `PreTrainedPolicy` and can be pushed/pulled from the Hub.
- **Processors** — small composable transforms between dataset → policy → robot.
- **Robots** (real) — hardware abstraction for recording and running policies. Sim `Envs` were trimmed to the minimum needed for type plumbing (no benchmark suites).
- **CLI** — `lerobot-record`, `lerobot-train`, `lerobot-rollout`, `lerobot-teleoperate`, `lerobot-calibrate`, `lerobot-find-port`, `lerobot-setup-motors`, `lerobot-replay`.

See [`AGENTS.md`](./AGENTS.md) for repo architecture.

---

## 3. Quickstart paths (pick one)

### Path A — "I have an SO-101 and want my first trained SmolVLA policy"

Go to §4 (SO-101 end-to-end), then §5 (data tips), then §7 (how long), then §8 (eval).

### Path B — "No hardware, I want to train on an existing dataset"

Skip §4. Pick a duration in §7, then run `lerobot-train` per §4.9 with a Hub `--dataset.repo_id`. Finish with §8.1 (real-robot eval) once you have hardware, or the inference smoke-test in `make test-smolvla-ete-eval` to sanity-check a checkpoint without hardware.

### Path C — "I just want to understand the codebase"

Read §2 above, then `AGENTS.md` "Architecture", then open `src/lerobot/policies/smolvla/` and `src/lerobot/datasets/lerobot_dataset.py` as canonical examples.

---

## 4. SO-101 end-to-end cheat-sheet

Full details in [`docs/source/so101.mdx`](./docs/source/so101.mdx) and [`docs/source/il_robots.mdx`](./docs/source/il_robots.mdx). Minimum commands in order. Confirm arms are assembled + powered before issuing.

**4.1 Install**

```bash
# uv (recommended — see AGENTS.md and CLAUDE.md)
uv sync --locked --extra feetech          # SO-100/SO-101 motor stack
# uv sync --locked --extra all            # everything
# uv sync --locked --extra smolvla        # add SmolVLA deps

# pip (alternative, e.g. when not working from source)
# pip install 'lerobot[feetech]'
# pip install 'lerobot[all]'
# pip install 'lerobot[smolvla]'

git lfs install && git lfs pull
hf auth login                             # required to push datasets/policies
```

**4.2 Find USB ports** — run once per arm, unplug when prompted.

```bash
lerobot-find-port
```

macOS: `/dev/tty.usbmodem...`; Linux: `/dev/ttyACM0` (may need `sudo chmod 666 /dev/ttyACM0`).

**4.3 Setup motor IDs & baudrate** (one-time, per arm)

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=<FOLLOWER_PORT>
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>
```

**4.4 Calibrate** — center all joints, press Enter, sweep each joint through its full range. The `id` is the calibration key — reuse it everywhere.

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>   --teleop.id=my_leader
```

**4.5 Teleoperate** (sanity check, no recording)

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

> **Feetech timeout / comms error on SO-100 / SO-101?** Before touching software, check the **red motor LEDs** on the daisy chain.
>
> - **All steady red, gripper → base chain** → wiring OK.
> - **One or more motors dark / chain stops mid-way** → wiring issue: reseat the 3-pin cables, check the controller-board power supply, and make sure each motor is fully clicked in.
> - **LEDs blinking** → the motor is in an **error state**: usually overload (forcing a joint past its limit) **or wrong power supply voltage**. SO-100 / SO-101 ship in two variants — a **5 V / 7.4 V** build and a **12 V** build — they are NOT interchangeable. Using a 12 V PSU on a 5 V / 7.4 V arm (or vice-versa) will trip this error; confirm your motor variant before powering up.
>
> Most "timeout" errors are physical, not code.

**4.6 Record a dataset** — keys: **→** next, **←** redo, **ESC** finish & upload.

```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')

lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/my_task \
  --dataset.single_task="<describe the task in one sentence>" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --display_data=true
```

**4.7 Visualize** — **always** do this before training. Look for missing frames, camera blur, unreachable targets, inconsistent object positions.
After upload: https://huggingface.co/spaces/lerobot/visualize_dataset → paste `${HF_USER}/my_task`. Works for **any LeRobot-formatted Hub dataset** — use it to scout other datasets, inspect episode quality, or debug your own data before retraining.

**4.8 Replay an episode** (sanity check)

```bash
lerobot-replay --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --dataset.repo_id=${HF_USER}/my_task --dataset.episode=0
```

**4.9 Train.** Apple silicon: `--policy.device=mps`. No local GPU? Add `--job.target=<flavor>` (e.g. `a10g-small`, list them with `hf jobs hardware`) to run on Hugging Face Jobs instead. See §7 for duration.

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/my_task \
  --policy.type=smolvla \
  --policy.device=cuda \
  --output_dir=outputs/train/smolvla_my_task \
  --job_name=smolvla_my_task \
  --batch_size=8 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/smolvla_my_task
```

**4.10 Evaluate on the real robot** — compare success rate to a teleoperated baseline.

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/eval_my_task \
  --dataset.single_task="<same task description as training>" \
  --dataset.num_episodes=10 \
  --policy.path=${HF_USER}/smolvla_my_task
```

---

## 5. Data collection tips (beginner → reliable policy)

Good data beats clever models. Adopt these defaults and deviate only with evidence.

### 5.1 Setup & ergonomics

- **Fix the rig and cameras** before touching the software. If the rig vibrates or the operator gets frustrated, fix that first — more bad data won't help.
- **Lighting matters more than resolution.** Diffuse, consistent light. Avoid moving shadows.
- **"Can you do the task from the camera view alone?"** If no, your cameras are wrong. Fix before recording.
- Enable **action interpolation** for rollouts when available for smoother trajectories.

### 5.2 Practice before you record

- Do 5–10 demos without recording. Build a deliberate, repeatable strategy.
- Hesitant or inconsistent demos teach the model hesitation.

### 5.3 Quality over speed

Deliberate, high-quality execution beats fast sloppy runs. Optimize for speed only **after** strategy is dialed in — never trade quality for it.

### 5.4 Consistency within and across episodes

Same grasp, approach vector, and timing. Coherent strategies are much easier to learn than wildly varying movements.

### 5.5 Start small, then extend (the golden rule)

- **First 50 episodes = constrained version** of the task: one object, fixed position, fixed camera setup, one operator.
- Train a quick SmolVLA run. See what fails.
- **Then add diversity** along one axis at a time: more positions → more lighting → more objects → more operators.
- Don't try to collect the "perfect dataset" on day one. Iterate.

### 5.6 Policy: SmolVLA

This fork only ships SmolVLA (language-conditioned, multi-task, small VLA — needs **12–16 GB VRAM** with defaults; see §6). Unfreezing the vision encoder (see §7.6) is a big win once you have a working baseline.

### 5.7 Recommended defaults for your first task

| Setting          | Value                                                                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Episodes         | **50** to start, scale to 100–300 after first training                                                                                                |
| Episode length   | 20–45 s (shorter is fine for grasp/place)                                                                                                             |
| Reset time       | 10 s                                                                                                                                                  |
| FPS              | 30                                                                                                                                                    |
| Cameras          | **2 cameras recommended**: 1 fixed front + 1 wrist. Multi-view often outperforms single-view. A single fixed camera also works to keep things simple. |
| Task description | Short, specific, action-phrased sentence                                                                                                              |

### 5.8 Troubleshooting signal

- Policy fails at one specific stage → record 10–20 more episodes **targeting that stage**.
- Policy flaps / oscillates → likely inconsistent demos, or need more training; re-record worst episodes (use **←** to redo).
- Policy ignores the object → camera framing or lighting issue, not a model issue.

See also: [What makes a good dataset](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset).

---

## 6. Do I have enough GPU for SmolVLA?

SmolVLA is the only policy in this fork, so there's no policy choice to make — just check whether the hardware can run it.

### 6.1 Profiling snapshot (indicative)

Typically trains for **5–10 epochs** (see §7).

> **Human-facing version:** the [Compute Hardware Guide](./docs/source/hardware_guide.mdx) has a cloud-GPU tier guide and a Hugging Face Jobs pointer.

| Policy    | Batch | Update (ms) | Peak GPU mem (GB) | Notes                                                                       |
| --------- | ----: | ----------: | -----------------: | ---------------------------------------------------------------------------- |
| `smolvla` |     1 |       357.8 |               3.93 | **Unfreeze vision encoder for big gains** (see §7.6); measured with frozen encoder. |

**Critical caveats:**

- **Optimizer:** measured with **SGD**. LeRobot's default is **AdamW**, which keeps extra optimizer state → **peak memory will be noticeably higher** with the default.
- **Batch size:** memory scales roughly linearly with batch; use gradient accumulation if VRAM-constrained (see §7.4).

### 6.2 Decision rules

- **< 8 GB VRAM (laptop, 3060, M-series Mac):** → feasible only with small batch + gradient accumulation, frozen vision encoder.
- **12–16 GB VRAM (4070/4080, A4000):** → SmolVLA with defaults.
- **24+ GB VRAM (3090/4090/A5000):** → unfreeze the vision encoder (§7.6) for better results.
- **80 GB (A100/H100):** → healthy batch, unfrozen vision encoder, comfortable headroom.
- **CPU only:** → don't train here. Use Google Colab (see [`docs/source/notebooks.mdx`](./docs/source/notebooks.mdx)) or a rented GPU.

---

## 7. How long should I train?

Robotics imitation learning usually converges in a **few epochs over the dataset**, not hundreds of thousands of raw steps. Think **epochs first**, then translate to steps.

### 7.1 Rule of thumb

- **Typical total: 5–10 epochs.** Start at 5, eval, then decide if more helps.
- Very small datasets (< 30 episodes) may want slightly more epochs — but first, **collect more data**.
- VLAs with a pretrained vision backbone typically need **fewer** epochs than training from scratch.

### 7.2 Steps ↔ epochs conversion

```
total_frames     = sum of frames over all episodes      # e.g. 50 eps × 30 fps × 30 s ≈ 45,000
steps_per_epoch  = ceil(total_frames / batch_size)
total_steps      = epochs × steps_per_epoch
```

Examples for `--batch_size=8`:

| Dataset size            |  Frames | Steps / epoch | 5 epochs | 10 epochs |
| ----------------------- | ------: | ------------: | -------: | --------: |
| 50 eps × 30 s @ 30 fps  |  45,000 |        ~5,625 |      28k |       56k |
| 100 eps × 30 s @ 30 fps |  90,000 |       ~11,250 |      56k |      113k |
| 300 eps × 30 s @ 30 fps | 270,000 |       ~33,750 |     169k |      338k |

Pass the resulting total with `--steps=<N>`; eval at intermediate checkpoints (`outputs/train/.../checkpoints/`).

### 7.3 SmolVLA starting point (single-task, ~50 episodes)

| Policy    | Batch | Steps (first run) | Notes                             |
| --------- | ----: | -----------------: | ---------------------------------- |
| `smolvla` |   4–8 |           30k–80k | Pretrained VLM → converges fast.   |

### 7.4 Batch size guidance

- **Bigger batch is preferable** for stable gradients on teleop data.
- If GPU memory is the bottleneck, use **gradient accumulation** to raise _effective_ batch without raising peak memory.
- Scale **learning rate** gently with batch; most LeRobot defaults work fine for a 2–4× batch change.

### 7.5 Scale LR schedule & checkpoints with `--steps`

LeRobot's default schedulers (e.g. SmolVLA's cosine decay) use `scheduler_decay_steps=30_000`, which is sized for long training runs. When you shorten training (e.g. 5k–10k steps on a small dataset), **scale the scheduler down to match** — otherwise the LR stays near the peak and never decays. Same for checkpoint frequency.

```bash
lerobot-train ... \
  --steps=5000 \
  --policy.scheduler_decay_steps=5000 \
  --save_freq=5000
```

Rule of thumb: set `scheduler_decay_steps ≈ steps`, and `save_freq` to whatever granularity you want for eval (e.g. every 1k–5k steps). Match `scheduler_warmup_steps` proportionally if your run is very short.

### 7.6 SmolVLA: unfreeze the vision encoder for real gains

SmolVLA ships with `freeze_vision_encoder=True`. Unfreezing usually **improves performance substantially** on specialized tasks, at the cost of more VRAM and slower steps. Enable with:

```bash
lerobot-train ... --policy.type=smolvla \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
```

### 7.7 Signals to stop / keep going

- Train loss plateaus → stop, save a Hub checkpoint.
- Train loss still dropping and you're under 10 epochs → keep going.

---

## 8. Evaluation

### 8.1 Real-robot eval (SO-101, etc.)

Reuse `lerobot-record` with `--policy.path` to run the trained policy on-robot and save the run as an eval dataset. Convention: prefix the dataset with `eval_`.

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/eval_my_task \
  --dataset.single_task="<same task description used during training>" \
  --dataset.num_episodes=10 \
  --policy.path=${HF_USER}/smolvla_my_task
```

Report success rate across episodes. Compare to a teleoperated baseline and to an earlier checkpoint to catch regressions.

### 8.2 No-hardware inference smoke test

To sanity-check a checkpoint without a robot, load it and run a real `select_action()` forward pass on a dataset frame — see `utils/smolvla_inference_smoke_test.py` (used by `make test-smolvla-ete-eval`). This only checks that inference runs and produces a valid action shape, not task success.

### 8.3 Target success rates

Single-task grasp-and-place with 50 clean episodes: expect **> 70% success** on the training configuration. Less → data problem (see §5), not model problem. Expect a drop when generalizing to new positions — scale episodes or diversity to recover.

---

## 9. Further reading & resources

- **Getting started:** [`installation.mdx`](./docs/source/installation.mdx) · [`il_robots.mdx`](./docs/source/il_robots.mdx) · [What makes a good dataset](https://huggingface.co/blog/lerobot-datasets)
- **SmolVLA & hardware docs:** browse [`docs/source/*.mdx`](./docs/source/) (`smolvla.mdx`, hardware guides, advanced training).
- **Community:** [Discord](https://discord.com/invite/s3KuuzsPFb) · [Hub `LeRobot` tag](https://huggingface.co/datasets?other=LeRobot) · [Dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)

> Keep this file current. If you learn a rule that would prevent a class of user mistakes, add it here and in [`AGENTS.md`](./AGENTS.md).
