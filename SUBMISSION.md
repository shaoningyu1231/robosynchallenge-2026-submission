# Submission: SmolVLA multi-task

> **Status: skeleton.** Items marked **TODO** must be filled in and verified
> before this repository is submitted. The last valid submission is the one
> evaluated, so only a version that passed the checks at the end is submitted.

## Run

Setup and the one standard evaluation command are in
[`policy/smolvla_multitask/README.md`](policy/smolvla_multitask/README.md):

```bash
bash policy/smolvla_multitask/setup_lerobot.sh
bash policy/smolvla_multitask/eval.sh <task_name> <setting> <gpu_id> [--key value ...]
```

## Checkpoints

Every task uses the checkpoint below unless it has its own row. Configured in
`policy/smolvla_multitask/deploy_policy.yml`; all are public on Hugging Face.

| Task | Hugging Face checkpoint | Revision |
|---|---|---|
| all ten (default) | **TODO** (multi-task model, not yet published) | **TODO** |

## Base model

SmolVLA as built by LeRobot `b6ec0060` with `--policy.type=smolvla`: the
vision-language backbone is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at
revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, truncated to 16 layers with
the vision encoder frozen; the action expert is trained from scratch. No
`smolvla_base` or other robot-pretrained weights. These fields match the
`train_config.json` of the official `SmolVLA_sim_click_bell` checkpoint.

The backbone revision is enforced at evaluation, not only documented: the `vlm`
entry of `deploy_policy.yml` pins it, and the worker loads it from that local
snapshot with the Hub offline.

## Data

Only the official simulation datasets, 1000 episodes per task, at these revisions:

| Dataset | Revision |
|---|---|
| RoboSynChallenge/cobotmagic_Sim_table_rearrangement | `ca14fdfbb9d39914f09b343e072618462bea97b8` |
| RoboSynChallenge/cobotmagic_Sim_click_bell | `dd1f0de495cfec3240beaf2714f239168d5b3fae` |
| RoboSynChallenge/cobotmagic_Sim_water_pouring | `932a287657983c41465d6c9baee5dd478ef61b90` |
| RoboSynChallenge/cobotmagic_Sim_handle_basket | `c40819caabfea75e0724307845ddf5ee7f15ae8a` |
| RoboSynChallenge/cobotmagic_Sim_items_handover | `20a88bf0bc3fe3a1ae58ad97ed60a06c3431684a` |
| RoboSynChallenge/cobotmagic_Sim_drawer_open_place | `6fdf6ee1ed74bbbbc0216d507c8a19c8994a15c3` |
| RoboSynChallenge/cobotmagic_Sim_mixer_operating | `fe5428c644932747b1109da98c6dc79a1caabd51` |
| RoboSynChallenge/cobotmagic_Sim_item_assembly | `033856dc40f698d9b4c82b87e00c8d171d92b925` |
| RoboSynChallenge/cobotmagic_Sim_manipulate_pipette | `168885aec78256de14e0c130ea12a6ba43e714a7` |
| RoboSynChallenge/cobotmagic_Sim_sample_loading | `67208f9ba05f2f29a77b9c5e7b64acf26c3b3c44` |

No real-robot data, no self-generated data, no external datasets.

Before merging the ten datasets into one (10000 episodes, 2,808,794 frames, 25 fps):

- feature and joint names unified (`observation.state`, `observation.images.cam_high`
  / `cam_left_wrist` / `cam_right_wrist`, `LEFT_JOINT1..7`, `RIGHT_JOINT1..7`)
- only `action`, `observation.state`, `observation.qvel`, `observation.qf` and the
  three cameras kept; object pose, contact and end-effector pose features dropped
  (the evaluation does not provide them)
- `table_rearrangement` gripper actions (dimensions 6 and 13) multiplied by 20,
  from metres (0–0.05) to the 0–1 opening fraction every other task uses
- joint columns stored as float32 vectors of length 14 (some tasks stored float64)
- converted from LeRobot v2.1 to v3.0

Every task keeps its official language instruction verbatim.
**TODO**: decide whether the dataset scripts ship in this repository or are linked.

## Training

`lerobot-train` through `policy/smolvla/finetune.sh`, on one RTX 5090:

| | |
|---|---|
| steps / batch size | 200,000 / 32 |
| optimizer | AdamW, peak lr 1e-4 |
| schedule | warmup 4,000 steps, cosine decay to 2.5e-6 at 120,000, constant after |
| seed | 1000 |
| image augmentation | off |

The official checkpoints use 50,000 steps, warmup 1,000 and decay 30,000; the
schedule here is the same curve stretched four times. Sampling is per frame, so
tasks are seen in proportion to their frame counts (`click_bell` 2.6%,
`drawer_open_place` 15.1%).

## Changes to upstream

Based on RoboSynChallenge `cccbbc99235add09c9bb41f0eef14f4ed0e89906`. All changes
are committed here; nothing needs to be patched at evaluation time.

- `policy/smolvla_multitask/` — the submitted policy (new)
- `policy/smolvla/deploy_policy.py`, `smolvla_worker.py` — `eval()` returns the four
  values `scripts/eval_policy.py` unpacks; camera frames cast to float32 before
  the policy; fallback when LeRobot lacks the processor converters
- `policy/smolvla/finetune.sh` — empty `rename_map` by default (required with
  `--policy.type` on current LeRobot)
- `scripts/eval_policy.py` — drains the simulator cleanup queue after
  `env.close()`, and writes `evaluation_metrics.json` before teardown so a crash
  in teardown cannot lose completed episodes

## Verified

**TODO** — record for the submitted commit, from a fresh clone of this repository:

- [ ] empty Hugging Face cache, not logged in: checkpoint downloads, 1 task × 3 episodes
- [ ] `--checkpoint_dir <local dir>`: 1 task × 3 episodes
- [ ] exit code 0, `evaluation_metrics.json` present with the right episode count
- [ ] metrics list accepted seeds (`episodes[].seed`) and skipped seeds
- [ ] no leftover processes or GPU memory after the run
