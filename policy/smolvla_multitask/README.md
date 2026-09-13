# SmolVLA multi-task policy

One SmolVLA model trained on all ten RoboSynChallenge simulation tasks, evaluated
through the standard `scripts/eval_policy.py`. Individual tasks can be pointed at
other checkpoints in `deploy_policy.yml`. Submission details (checkpoints, data,
training recipe, changes to upstream): [`SUBMISSION.md`](../../SUBMISSION.md).

## Setup

1. Install the simulator environment as in the repository README
   (EmbodiChain v0.2.4, Python 3.11, `EmbodiChain/.venv` next to this repository).
2. Build the SmolVLA worker environment (Python 3.12, LeRobot `b6ec0060`):

   ```bash
   bash policy/smolvla_multitask/setup_lerobot.sh
   ```

   The two environments must stay separate: the worker needs numpy 2, the
   simulator's binaries need numpy 1. They communicate only through
   `smolvla_worker.py`.

## Evaluate

The one standard command, from the repository root:

```bash
bash policy/smolvla_multitask/eval.sh <task_name> <setting> <gpu_id> [--key value ...]
```

For example, 100 episodes of `click_bell` with the random setting on GPU 0:

```bash
bash policy/smolvla_multitask/eval.sh click_bell random 0 --max_episodes 100 --seed 0
```

- `task_name`: `table_rearrangement`, `click_bell`, `water_pouring`, `handle_basket`,
  `items_handover`, `drawer_open_place`, `mixer_operating`, `item_assembly`,
  `manipulate_pipette`, `sample_loading`
- `setting`: `random` or `clear`
- any `--key value` overrides that key of `deploy_policy.yml`

Each run writes
`eval_result/<task_name>/smolvla_multitask/<setting>/None/<model_name>/<timestamp>/evaluation_metrics.json`
(`--eval_result_dir` changes the root). Besides the summary it records
`config.checkpoint_path` (the exact weights, e.g. `hf://org/name@<revision>`),
every accepted episode seed in `episodes[].seed`, and every seed the expert
feasibility filter skipped in `feasibility_filter.skipped_seeds`.

## Which checkpoint is used

Decided per task, in this order (`checkpoints.py`):

1. `--checkpoint_dir <dir>`, if passed: a local checkpoint directory, or a LeRobot
   step directory containing `pretrained_model/`
2. `checkpoints.tasks.<task_name>` in `deploy_policy.yml`
3. `checkpoints.default` in `deploy_policy.yml`
4. otherwise the evaluation stops with an error

There is no fallback to the official checkpoints. Hugging Face entries give
`repo_id` and the full 40-character commit hash as `revision`; they are
downloaded anonymously into the Hugging Face cache on first use, so they must be
public. Downloads honour `HF_ENDPOINT`, `HF_HOME` and `HF_HUB_OFFLINE`.

## Baseline check

`test_configs/official_baseline.yml` maps every task to the official
per-task SmolVLA checkpoint at the revision the leaderboard evaluated. It exists
only to check this adapter against a known baseline and is never the default:

```bash
SMOLVLA_DEPLOY_CONFIG=policy/smolvla_multitask/test_configs/official_baseline.yml \
    bash policy/smolvla_multitask/eval.sh click_bell random 0 --max_episodes 3
```

## Tests

```bash
python -m unittest tests.test_smolvla_multitask_checkpoints -v
```

No simulator, torch or network needed; the tests of the shipped YAML files need PyYAML.
