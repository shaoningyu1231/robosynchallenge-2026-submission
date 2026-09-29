# Submission: official SmolVLA checkpoints, one per task, pinned

Every task is evaluated with the organisers' own released single-task SmolVLA
checkpoint, each pinned to a full Hugging Face commit. This submission contains
no self-trained weights and references none: no checkpoint of ours is included,
downloaded or linked, and no weights are redistributed here. What it adds is the
evaluation adapter that loads those checkpoints reproducibly (below).

## Run

Setup and the one standard evaluation command are in
[`policy/smolvla_multitask/README.md`](policy/smolvla_multitask/README.md):

```bash
bash policy/smolvla_multitask/setup_lerobot.sh
bash policy/smolvla_multitask/eval.sh <task_name> <setting> <gpu_id> [--key value ...]
```

The entrypoints are `policy/smolvla_multitask/eval.sh`,
`policy/smolvla_multitask/deploy_policy.yml` and
`policy/smolvla_multitask/deploy_policy.py`.

## Checkpoints

From `checkpoints.tasks` and `vlm` in
[`policy/smolvla_multitask/deploy_policy.yml`](policy/smolvla_multitask/deploy_policy.yml),
which is what the evaluation reads; this table is generated from that file and
checked against it.

| Task | Hugging Face checkpoint | Revision | Mode |
|---|---|---|---|
| `items_handover` | `RoboSynChallenge/SmolVLA_sim_items_handover` | `fabe9462853965228e90e34bd517aa4e43bad323` | `legacy` |
| `table_rearrangement` | `RoboSynChallenge/SmolVLA_sim_table_rearrangement` | `913b29a7136876108a330aead1855309f985cbd8` | `legacy` |
| `click_bell` | `RoboSynChallenge/SmolVLA_sim_click_bell` | `27965b7e2432f5a3084f2c60a04bff7c5a47b853` | `legacy` |
| `water_pouring` | `RoboSynChallenge/SmolVLA_sim_water_pouring` | `2bf5d9c2d8d0b5820ba5c9fe6284c5ff8647561f` | `legacy` |
| `handle_basket` | `RoboSynChallenge/SmolVLA_sim_handle_basket` | `0de9a93f124bd4ebe2fbd0d572ce3133806b82a9` | `legacy` |
| `drawer_open_place` | `RoboSynChallenge/SmolVLA_sim_drawer_open_place` | `c0088d84a568f93fb4401aabafcc41cf643efcdd` | `legacy` |
| `mixer_operating` | `RoboSynChallenge/SmolVLA_sim_mixer_operating` | `42311ef3e795079caefee7a28f05c967caa870ac` | `legacy` |
| `item_assembly` | `RoboSynChallenge/SmolVLA_sim_item_assembly` | `db32020164ed2c7264804a5eb036a38a07d879ca` | `legacy` |
| `manipulate_pipette` | `RoboSynChallenge/SmolVLA_sim_manipulate_pipette` | `dd0e80bc7d5fa5081e1fa07f1b7115bad6dbffa0` | `legacy` |
| `sample_loading` | `RoboSynChallenge/SmolVLA_sim_sample_loading` | `2a4f41ab3be8e5f544c890d6054bc8de3be5c53b` | `legacy` |
| VLM backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | — |

- **Revision**: every entry is a full 40-character commit. The adapter downloads
  exactly that commit, without logging in, and refuses a branch name or a short
  hash.
- **Mode** `legacy`: the normalization the official checkpoints were trained
  with, applied as the organisers' own evaluation applies it. Each task declares
  its mode beside its checkpoint (`policy/smolvla_multitask/checkpoints.py`).
- There is no default entry: a task missing from the table stops the evaluation
  instead of falling back to any other model.
- The VLM backbone is pinned as well: a SmolVLA checkpoint names its backbone by
  repo id only, so without the pin an empty cache would build the model on that
  repository's current main.

## Data and training

None. This submission trains nothing and uses no data of its own; the
checkpoints above are the organisers' releases, unchanged.

## Changes to upstream

Based on RoboSynChallenge `cccbbc99235add09c9bb41f0eef14f4ed0e89906`. All changes
are committed here; nothing needs to be patched at evaluation time.

- `policy/smolvla_multitask/` — the submitted policy adapter (new): checkpoint
  resolution and pinning (`checkpoints.py`, `fetch_checkpoint.py`), the worker,
  per-episode flow-matching noise seeding (`noise_seed.py`), normalization modes
  (`task_norm.py`), the pinned LeRobot environment (`setup_lerobot.sh`,
  `requirements.lock`) and `eval.sh`
- `policy/smolvla/deploy_policy.py`, `smolvla_worker.py` — `eval()` returns the four
  values `scripts/eval_policy.py` unpacks; camera frames cast to float32 before
  the policy; fallback when LeRobot lacks the processor converters
- `policy/smolvla/deploy_policy.yml` — two options, both off by default so the
  official behavior is kept (`smolvla_check_success_each_step: false`,
  `smolvla_replan_steps: null`)
- `policy/smolvla/finetune.sh` — empty `rename_map` by default (required with
  `--policy.type` on current LeRobot); not used by this submission
- `.gitignore` — the LeRobot checkout and venv that `setup_lerobot.sh` builds
- `scripts/eval_policy.py` — drains the simulator cleanup queue after
  `env.close()`, and writes `evaluation_metrics.json` before teardown so a crash
  in teardown cannot lose completed episodes; optional frozen episode seeds for
  our own comparisons (off by default: `eval_episode_seeds: null` keeps the
  organisers' protocol)
- `tests/` — tests for all of the above

## Verified

Final acceptance, 2026-09-26, on commit `f10ba96861af89f1dfc24eb1afdae51bf6ca4e2c`
(tree `1a8c995617cb4470c8fe046e97017b7bf7d5af0e`; `deploy_policy.yml` sha256
`32c052abcc21a4121bbee41a57beaac224e1048897dd210fd271b3531f2abee9`). All passed:

- [x] fresh clone of the branch, tree identical to the commit, clone clean
- [x] full test suite inside the clean clone (123 passed)
- [x] each of the ten tasks: one real reset and inference, each loading its own pinned revision
- [x] `evaluation_metrics.json` checked field by field: task, checkpoint, mode, revision
- [x] four invalid configurations rejected before the worker starts (unknown task; a
      required-mode declaration on an official checkpoint; a required checkpoint whose
      statistics digest was changed; a checkpoint carrying statistics declared legacy)
- [x] empty Hugging Face cache, not logged in, this configuration itself: downloads and evaluates
- [x] local checkpoint directory (`--checkpoint_dir`)
- [x] two runs accept and skip the same episode seeds; no leftover processes; GPU memory back to zero

The commit that carries this file changes `SUBMISSION.md` only; the code, the
configuration and the entrypoints are byte-identical to the accepted commit above.
