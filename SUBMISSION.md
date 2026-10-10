# Submission: five tasks on this project's pi0 checkpoints, five on the official SmolVLA, all pinned

Five tasks are evaluated with checkpoints this project trained (pi0 / pi05 with
LoRA, fine-tuned on the challenge's own simulation datasets) and published as
public, non-gated Hugging Face repositories; the other five are evaluated with
the organisers' own released single-task SmolVLA checkpoints. Every entry is
pinned to a full 40-character Hugging Face commit and downloads **without
logging in**.

No weights are stored in this repository. The five pi0 entries are references to
published repositories, downloaded at evaluation time exactly like the official
ones. What this repository adds is the evaluation adapter and the pi0 backend
that loads them reproducibly (below).

The evidence behind the five pi0 entries is **not the same for all of them**, and
the table below says which is which. Three passed an independent test; two did
not have one and were adopted on the project owner's decision. Nothing here
claims a result for a checkpoint that did not earn it.

## Run

Setup and the one standard evaluation command are in
[`policy/smolvla_multitask/README.md`](policy/smolvla_multitask/README.md). The
SmolVLA worker environment and the pi0 (pi05) environment are built separately;
a task's entry decides which one runs it.

```bash
bash policy/smolvla_multitask/setup_lerobot.sh    # the SmolVLA worker
bash policy/smolvla_multitask/setup_pi0.sh        # the pi05 environment, for the five pi0 tasks
bash policy/smolvla_multitask/eval.sh <task_name> <setting> <gpu_id> [--key value ...]
```

The evaluation command is unchanged: `backend: pi0` is handled behind it. The
entrypoints are `policy/smolvla_multitask/eval.sh`,
`policy/smolvla_multitask/deploy_policy.yml` and
`policy/smolvla_multitask/deploy_policy.py`.

`setup_pi0.sh` builds a second environment (JAX; roughly 12 GB) and fetches the
PaliGemma tokenizer from its original source, accepting only the pinned sha256.
If that environment cannot be built on the evaluating machine, the five pi0 tasks
cannot run; that is the largest single risk in this submission, and it is stated
here rather than discovered at evaluation time.

### What `setup_pi0.sh` assumes about the machine

Building the pi05 environment from a fresh clone on our own evaluation box turned
up four prerequisites. All four come from `policy/pi05/` as upstream ships it, not
from anything this submission changed, and `setup_pi0.sh` and `policy/pi05/uv.lock`
are byte-identical to upstream's. They are listed with the error each one produces
so they can be recognised rather than diagnosed.

1. **A CPython 3.11.** `policy/pi05/pyproject.toml` requires `>=3.11,<3.12`. If the
   system Python is not 3.11, uv downloads a managed interpreter, which needs
   access to the python-build-standalone release assets on GitHub. An existing
   3.11 can be supplied instead with `UV_PYTHON=/path/to/python3.11`.
2. **An EmbodiChain checkout beside this repository.** `policy/pi05/uv.lock` and
   `pyproject.toml` pin `embodichain` as an editable path dependency at
   `../../../EmbodiChain`, which is the layout the official installation document
   prescribes: `RoboSynChallenge_ws/{EmbodiChain,RoboSynChallenge}`. Without the
   sibling, `uv sync --frozen` stops with
   `Distribution not found at: file:///<parent>/EmbodiChain`. The `EMBODICHAIN_ROOT`
   variable does not help here: it only selects the simulator's Python, while the
   path in the lock is resolved by uv.
3. **Reachable `github.com`.** The lock pins two dependencies to commits —
   `lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` and
   `dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a`, both upstream openpi's. Without
   access, `uv sync` stops at `unable to access 'https://github.com/huggingface/lerobot/'`.
   Either commit may also be served from a local mirror through git's `insteadOf`;
   the 40-character commit is what pins the content either way.
4. **Patience on a slow link.** `uv sync --frozen` fetches every wheel from the URL
   written in the lock (`files.pythonhosted.org`), not from a configured index
   mirror, and runs them concurrently. On a thin connection a single large wheel can
   exceed uv's 30 s default and the build stops with
   `Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT`.
   Raising `UV_HTTP_TIMEOUT` is enough; nothing in the submission needs to change.

## Checkpoints

From `checkpoints.tasks` and `vlm` in
[`policy/smolvla_multitask/deploy_policy.yml`](policy/smolvla_multitask/deploy_policy.yml),
which is what the evaluation reads; this table is generated from that file and
checked against it.

| Task | Hugging Face checkpoint | Revision | Mode |
|---|---|---|---|
| `items_handover` | `kolmann1231/pi0-robosyn-handover-24k` | `219b8ccb6a9a3f1f64beba698c0e41637d50517e` | `pi0` |
| `table_rearrangement` | `RoboSynChallenge/SmolVLA_sim_table_rearrangement` | `913b29a7136876108a330aead1855309f985cbd8` | `legacy` |
| `click_bell` | `RoboSynChallenge/SmolVLA_sim_click_bell` | `27965b7e2432f5a3084f2c60a04bff7c5a47b853` | `legacy` |
| `water_pouring` | `kolmann1231/pi0-robosyn-water-16k` | `3b6487599499eecb29acf25c3e54b5f73193b96e` | `pi0` |
| `handle_basket` | `RoboSynChallenge/SmolVLA_sim_handle_basket` | `0de9a93f124bd4ebe2fbd0d572ce3133806b82a9` | `legacy` |
| `drawer_open_place` | `kolmann1231/pi0-robosyn-drawer-16k` | `2c1edf66fe0b2d4d3bec9caa9c7a9d9a7284117b` | `pi0` |
| `mixer_operating` | `kolmann1231/pi0-robosyn-mixer-16k` | `940fc3b587a739e27c9b16dca57de35ce00cf04b` | `pi0` |
| `item_assembly` | `RoboSynChallenge/SmolVLA_sim_item_assembly` | `db32020164ed2c7264804a5eb036a38a07d879ca` | `legacy` |
| `manipulate_pipette` | `kolmann1231/pi0-robosyn-pipette-24k` | `7a56dc1b5ea33f5acf13cc6c7f8fc8bdff56a292` | `pi0` |
| `sample_loading` | `RoboSynChallenge/SmolVLA_sim_sample_loading` | `2a4f41ab3be8e5f544c890d6054bc8de3be5c53b` | `legacy` |
| VLM backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | — |

- **Revision**: every entry is a full 40-character commit. The adapter downloads
  exactly that commit, without logging in, and refuses a branch name or a short
  hash.
- **Mode** `legacy`: the normalization the official checkpoints were trained
  with, applied as the organisers' own evaluation applies it. **Mode** `pi0`: the
  task is served by the pi05 backend, which uses the normalizer published inside
  that checkpoint's own repository; the entry also names the openpi training
  config it must be loaded with (`pi0_config`). Each task declares its mode
  beside its checkpoint (`policy/smolvla_multitask/checkpoints.py`).
- There is no default entry: a task missing from the table stops the evaluation
  instead of falling back to any other model.
- The VLM backbone pin applies to the SmolVLA tasks: a SmolVLA checkpoint names
  its backbone by repo id only, so without the pin an empty cache would build the
  model on that repository's current main.

## What stands behind each pi0 entry

Each comparison below is this project's own measurement against the official
checkpoint for that task, on scenes drawn fresh for the test and used once, both
systems run on the same scenes. Each test used alpha = 0.05 on its own; **no
correction was made across tasks, so these must not be read as a 5% error rate
for the set as a whole.** Success rate is 75% of the competition score, so a
point of success rate is not a point of score.

| Task | Checkpoint | Evidence |
|---|---|---|
| `drawer_open_place` | pi0 16k | **Independent test passed**: 203/300 vs official 131/300, +24.0 points (95% interval +16.5 to +31.1), p = 1.1e-9 |
| `mixer_operating` | pi0 16k | **Independent test passed**: 253/300 vs official 178/300, +25.0 points (95% interval +18.5 to +31.3), p = 7.1e-13. The verdict is under the test plan as revised by its addendum 4, which settled how one episode with a joint-jump flag is counted; re-recording that episode — the official side's success — as a failure would make it 177/300 and +76, so the conclusion does not turn on it |
| `water_pouring` | pi0 16k | **Independent test passed**: 273/300 vs official 201/300, +24.0 points (95% interval +17.9 to +30.0), p = 1.7e-13 |
| `items_handover` | pi0 **24k** | **Development-pool selection, NOT independently validated.** On a 50-scene development pool it led the 16k by 4 (39 vs 35; 95% interval -4.3 to +20.2, p = 0.344), which did not reach the pre-registered threshold of 5, and the pre-registered verdict was to keep the 16k. It is submitted on the project owner's decision of 2026-10-09. The checkpoint that passed an independent test for this task is the **16k** (187/300 vs 95/300, +30.7 points, 95% interval +22.8 to +38.0, p = 3.9e-13) — that result belongs to the 16k and is **not** a result for the 24k submitted here |
| `manipulate_pipette` | pi0 **24k** | **Development-pool selection, NOT independently validated.** On a 50-scene development pool it led the official checkpoint by 6 (31 vs 25; 95% interval -5.1 to +28.1, p = 0.263 — an interval that does not exclude no improvement), and it led the 16k by 4 (31 vs 27, p = 0.481). Submitted on the project owner's decision of 2026-10-09. This project has never run an independent test for this task |

A development pool is reused across candidates and the candidate submitted was
chosen on it, so its margin is an optimistic estimate, not a measurement of the
improvement. Two independent tests for the 24k checkpoints are planned; if they
are run, their outcome will be recorded here as it comes out, and a test that
does not reach its threshold will be recorded as that — not as a pass.

## Data and training

The five pi0 checkpoints were trained by this project with LoRA on top of
`pi0_base` (Physical Intelligence / openpi), fine-tuned on the challenge's own
simulation datasets (`cobotmagic_Sim_<task>`). The five SmolVLA entries are the
organisers' releases, unchanged. The training code is in `policy/pi05/`; the
weights themselves are not in this repository.

## Licensing and redistribution: what is settled and what is not

This section states the position as it stands. **It is not a claim that
redistribution is fully cleared.**

Settled, and shipped inside each published pi0 repository:

- A copy of the current Gemma Terms of Use (`GEMMA_TERMS_OF_USE.txt`). It was
  extracted from the terms page (Last modified April 1, 2026, snapshot kept) and
  machine-checked to be character-for-character the terms text; a link appendix
  was added at the end, marked as ours and not part of the terms, because plain
  text drops the 28 hyperlink targets. The project owner reviewed it paragraph by
  paragraph against the page.
- A `NOTICE` carrying, verbatim, the sentence section 3.1 of those terms requires,
  plus a statement of what was modified.
- A model card stating the modification and carrying the section 3.2 use
  restrictions as terms binding on anyone downstream, and stating the evidence
  status of that checkpoint in the same words as the table above.

Not settled, and recorded as open:

- **`pi0_base` weights.** At the original source
  (`github.com/Physical-Intelligence/openpi`) there is an Apache-2.0 `LICENSE` and
  a `LICENSE_GEMMA.txt` (the 2024-02-21 version) side by side, and the README
  only says the checkpoint is downloaded from `gs://openpi-assets`. No separate
  licence statement for the weights was found. Third-party re-uploads that claim
  one are not the original source and are not relied on. **This is not taken as
  permission having been granted.**
- **Derivative weights of the challenge datasets.** The datasets carry no licence
  label on Hugging Face, and the organisers have not been asked. The rules allow
  a different Hugging Face checkpoint per task and require a private repository to
  invite the organisers, which shows participants are expected to train on these
  datasets and submit weights — but that is not written confirmation of
  redistribution terms. **Unresolved.**
- `policy/pi05/LICENSE_GEMMA.txt` in this repository is the upstream 2024-02-21
  version, which is still what upstream openpi carries. The current version
  accompanies the published weights; whether this repository should also carry it
  is a separate decision and has not been made.

The PaliGemma tokenizer is **not** packaged with the weights: the adapter
downloads it from its original source and accepts only the pinned sha256.

The five pi0 repositories are public and not gated, because the adapter
downloads anonymously (`token=False`) and a gated or private repository cannot be
downloaded that way even by an invited account.

## Changes to upstream

Based on RoboSynChallenge `cccbbc99235add09c9bb41f0eef14f4ed0e89906`. All changes
are committed here; nothing needs to be patched at evaluation time.

- `policy/smolvla_multitask/` — the submitted policy adapter (new): checkpoint
  resolution and pinning (`checkpoints.py`, `fetch_checkpoint.py`), the worker,
  per-episode flow-matching noise seeding (`noise_seed.py`), normalization modes
  (`task_norm.py`), the pinned LeRobot environment (`setup_lerobot.sh`,
  `requirements.lock`) and `eval.sh`
- `policy/smolvla_multitask/pi0_backend.py`, `pi0_io.py`, `pi0_serve.py`,
  `setup_pi0.sh` — the pi0 route (new): a task with `backend: pi0` starts a local
  pi05 server, which is torn down with the episode; the pi05 environment is built
  with `uv sync --frozen`, the client dependency pinned by hash, and the
  PaliGemma tokenizer fetched from its original source and accepted only at its
  pinned sha256 (each download writes its own temporary file and is moved in
  atomically; `PI0_TOKENIZER_OFFLINE=1` forbids any download)
- `policy/pi05/` — the training and inference tree the pi0 checkpoints were
  trained and evaluated with, including the named task configs the five entries
  reference
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

**The engineering acceptance of THIS submission has not been run yet.** Nothing
in this section may be read as a result for the five-pi0 configuration until it
is replaced by the acceptance's own record.

What has been accepted is the all-official submission: final acceptance
2026-09-26 on commit `f10ba96861af89f1dfc24eb1afdae51bf6ca4e2c`
(`deploy_policy.yml` sha256
`32c052abcc21a4121bbee41a57beaac224e1048897dd210fd271b3531f2abee9`), which
remains available as the tags `submission-safe-f10ba96` and
`submission-final-3a73cd5`. That acceptance covers the SmolVLA route and the five
entries that are still official; **it does not cover the pi0 route**, which it
never ran.

The pi0 route has so far been accepted only from local checkpoint directories
(drawer, items_handover, mixer_operating, water_pouring). Downloading a pi0
checkpoint from Hugging Face by pinned revision into an empty cache with no
credential has **not** yet been accepted end to end, and `manipulate_pipette` has
not been through a delivery acceptance at all. That is exactly what the pending
acceptance has to prove, for all ten tasks, from a fresh clone.
