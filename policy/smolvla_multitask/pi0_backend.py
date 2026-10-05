# ----------------------------------------------------------------------------
# pi0 backend: a task whose checkpoint entry says `backend: pi0`
# ----------------------------------------------------------------------------
#
# deploy_policy.get_model hands such a task here. The simulator side stays in the
# evaluator's environment; the model runs in policy/pi05's own environment
# (setup: policy/smolvla_multitask/setup_pi0.sh), behind a websocket on 127.0.0.1,
# started here and stopped when the evaluation ends:
#   - the server runs in its own process group; atexit sends it SIGTERM, then
#     SIGKILL after 30 s, and waits for it;
#   - the server also ends itself when this process is gone (pi0_serve.py's
#     parent watchdog), so even a killed evaluation leaves no model on the GPU;
#   - a server that exits or does not listen within START_TIMEOUT_S stops the
#     evaluation with the end of its log, and is cleaned up the same way.
#
# Behaviour per call is the research adapter's, under the rules the drawer
# checkpoint won its independent test with (research repository,
# experiments/B3_DRAWER_TEST_RESULT.txt):
#   - one inference every PI0_STEP = 10 environment steps; the first 10 of the
#     50 returned actions are executed, stopping at success or truncation;
#   - observation and action mapping: pi0_io.py (grippers trained as a [0, 1]
#     ratio, mapped onto the action space; arms passed through unchanged);
#   - arm values beyond the joint limits are sent as they are and clamped by the
#     engine, as for SmolVLA; they are counted per joint for the episode record;
#   - the flow-matching noise of every call is fixed by (pi0_noise_base, the
#     episode's environment seed, the call's index) -- NOISE_DERIVATION -- and the
#     server's digest of the noise it used must equal ours;
#   - a non-finite action stops the evaluation.

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from policy.smolvla_multitask import pi0_io as io

POLICY_DIR = Path(__file__).resolve().parent
REPO_ROOT = POLICY_DIR.parents[1]
PI05_DIR = REPO_ROOT / "policy" / "pi05"
SERVE_SCRIPT = POLICY_DIR / "pi0_serve.py"
CLIENT_SRC = PI05_DIR / "packages" / "openpi-client" / "src"
DEFAULT_CLIENT_DEPS = POLICY_DIR / ".pi0_client_deps"       # msgpack, installed by setup_pi0.sh
DEFAULT_DATA_HOME = POLICY_DIR / ".pi0_openpi_data"         # OPENPI_DATA_HOME of the server
TOKENIZER = "big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
# what a pi0 checkpoint cannot be served without (openpi's params and its normalizer)
REQUIRED_FILES = ("params/_METADATA", "params/manifest.ocdbt")
PI0_STEP = 10
START_TIMEOUT_S = 600
STOP_GRACE_S = 30
NOISE_KEY = "research/noise"
NOISE_DERIVATION = ("seed = int.from_bytes(sha256(f'pi0-noise:{base}:{env_seed}:{call}').digest()[:8], 'big') & (2**63 - 1); "
                    "noise = numpy.random.Generator(PCG64(seed)).standard_normal((action_horizon, action_dim), float32); "
                    "call = 0, 1, ... within the episode")


class Pi0Error(RuntimeError):
    pass


def call_noise_seed(base: int, env_seed: int, call: int) -> int:
    digest = hashlib.sha256(f"pi0-noise:{int(base)}:{int(env_seed)}:{int(call)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def call_noise(base: int, env_seed: int, call: int, shape) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(call_noise_seed(base, env_seed, call)))
    return rng.standard_normal(tuple(shape), dtype=np.float32)


def noise_digest(noise: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(noise, dtype=np.float32).tobytes()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_checkpoint(path: Path) -> Path:
    """The served directory: openpi params plus exactly one normalizer under assets/."""
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    norms = sorted((path / "assets").glob("*/*/norm_stats.json"))
    if missing or len(norms) != 1:
        raise Pi0Error(f"{path} is not a servable pi0 checkpoint: missing {missing}, normalizers {len(norms)} (need 1)")
    return path


def prepare_data_home(checkpoint_dir: Path, data_home: Path) -> str:
    """The PaliGemma tokenizer where openpi looks for it (OPENPI_DATA_HOME/big_vision/...), so the server does not need
    to reach Google Cloud Storage: taken from the checkpoint snapshot when it carries one, never trusted unhashed."""
    target = data_home / TOKENIZER
    if target.is_file() and sha256_file(target) == TOKENIZER_SHA256:
        return "present"
    shipped = checkpoint_dir / TOKENIZER
    if shipped.is_file():
        if sha256_file(shipped) != TOKENIZER_SHA256:
            raise Pi0Error(f"{shipped} has sha256 {sha256_file(shipped)}, expected {TOKENIZER_SHA256}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shipped, target)
        return "from the checkpoint snapshot"
    return "absent (openpi downloads it from gs://big_vision on first use)"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Pi0Server:
    """policy/pi05's server process for one checkpoint, in its own process group."""

    def __init__(self, python: str, config: str, checkpoint_dir: Path, gpu_id, data_home: Path, log_path: Path):
        self.python, self.config, self.checkpoint_dir = python, config, checkpoint_dir
        self.gpu_id, self.data_home, self.log_path = gpu_id, data_home, log_path
        self.port = None
        self.proc = None

    def start(self) -> None:
        if not Path(self.python).is_file():
            raise Pi0Error(f"no policy/pi05 environment at {self.python} (run policy/smolvla_multitask/setup_pi0.sh)")
        self.port = _free_port()
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "VIRTUAL_ENV")}
        env.update(CUDA_VISIBLE_DEVICES=str(self.gpu_id), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   OPENPI_DATA_HOME=str(self.data_home), PYTHONUNBUFFERED="1")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        cmd = [self.python, str(SERVE_SCRIPT), "--port", str(self.port), "policy:checkpoint",
               "--policy.config", self.config, "--policy.dir", str(self.checkpoint_dir)]
        self.proc = subprocess.Popen(cmd, cwd=PI05_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()
        atexit.register(self.stop)
        deadline = time.time() + START_TIMEOUT_S
        while time.time() < deadline:
            code = self.proc.poll()
            if code is not None:
                self.stop()
                raise Pi0Error(f"the pi0 server exited with {code} before listening; "
                               f"end of {self.log_path}:\n{self.log_tail()}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=2):
                    return
            except OSError:
                time.sleep(2)
        self.stop()
        raise Pi0Error(f"the pi0 server did not listen within {START_TIMEOUT_S} s; end of {self.log_path}:\n{self.log_tail()}")

    def log_tail(self, n: int = 30) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "(no log)"

    def stop(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
        if pgid is not None:
            for sig, wait_s in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, 10)):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait_s)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.proc = None


class Pi0Model:
    """What get_model returns for a pi0 task; the evaluator only passes it back to eval / reset / episode_metadata."""

    def __init__(self, client, server: Pi0Server, noise_base: int, metadata: dict):
        self.client, self.server, self.noise_base, self.metadata = client, server, noise_base, metadata
        self.prompt = None
        self.process_calls = 0
        self.server_metadata = client.get_server_metadata()
        spec = (self.server_metadata or {}).get("research_noise")
        if not spec or spec.get("key") != NOISE_KEY:
            raise Pi0Error("the pi0 server does not take the noise of every call (pi0_serve.py)")
        self.noise_shape = tuple(spec["shape"])
        self.reset()

    def reset(self) -> None:
        self.prompt = None
        self.record = {"inferences": 0, "round_trip_s": [], "server_infer_ms": [], "actions_executed": 0,
                       "gripper_values": 0, "gripper_outside_unit": 0, "arm_values": 0, "arm_outside_action_space": 0,
                       "arm_beyond_tol_by_dim": {}, "arm_excursion_max_by_dim": {}, "nonfinite_outputs": 0,
                       "includes_process_first_call": False, "env_seed": None, "noise_digests": []}

    def close(self) -> None:
        self.server.stop()


def _client(host: str, port: int, deps: Path):
    for path in (CLIENT_SRC, deps):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from openpi_client import msgpack_numpy as _mp
    from openpi_client import websocket_client_policy as _wcp
    import websockets.sync.client as _wsc

    class _Client(_wcp.WebsocketClientPolicy):
        # keepalive pings off: the server compiles on its first inference and does not answer pings meanwhile
        def _wait_for_server(self):
            while True:
                try:
                    conn = _wsc.connect(self._uri, compression=None, max_size=None, ping_interval=None)
                    return conn, _mp.unpackb(conn.recv())
                except ConnectionRefusedError:
                    time.sleep(2)

    return _Client(host=host, port=port)


def _noise_base(value) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Pi0Error(f"pi0_noise_base must be a non-negative integer, got {value!r}")
    return value


def get_model(usr_args: dict, source, materialize, python_bin: str) -> Pi0Model:
    """source: the task's CheckpointSource (backend pi0); materialize: checkpoints.materialize."""
    checkpoint_dir = check_checkpoint(materialize(source, python_bin, required=REQUIRED_FILES))
    noise_base = _noise_base(usr_args.get("pi0_noise_base"))
    data_home = Path(usr_args.get("pi0_openpi_data_home") or os.environ.get("PI0_OPENPI_DATA_HOME") or DEFAULT_DATA_HOME)
    tokenizer = prepare_data_home(checkpoint_dir, data_home)
    pi05_python = usr_args.get("pi0_python") or os.environ.get("PI0_PYTHON") or str(PI05_DIR / ".venv" / "bin" / "python")
    deps = Path(usr_args.get("pi0_client_deps") or os.environ.get("PI0_CLIENT_DEPS") or DEFAULT_CLIENT_DEPS)
    log_path = Path(usr_args.get("pi0_server_log") or (REPO_ROOT / "eval_result" / "pi0_server" /
                                                      f"{usr_args['task_name']}_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    norm = next((checkpoint_dir / "assets").glob("*/*/norm_stats.json"))
    print(f"pi0 for {usr_args['task_name']}: {source.describe()} (from {source.origin}) -> {checkpoint_dir}; "
          f"config {source.pi0_config}; normalizer {sha256_file(norm)[:16]}; tokenizer {tokenizer}; "
          f"server log {log_path}", flush=True)
    server = Pi0Server(pi05_python, source.pi0_config, checkpoint_dir, usr_args.get("gpu_id", 0), data_home, log_path)
    server.start()
    try:
        client = _client("127.0.0.1", server.port, deps)
        metadata = {"policy": {"backend": "pi0", "origin": source.origin, "repo_id": source.repo_id,
                               "revision": source.revision, "path": str(checkpoint_dir), "pi0_config": source.pi0_config,
                               "normalizer_sha256": sha256_file(norm)},
                    "policy_noise": {"base_seed": noise_base, "per_call": NOISE_DERIVATION}}
        model = Pi0Model(client, server, noise_base, metadata)
    except Exception:
        server.stop()
        raise
    usr_args["checkpoint_dir"] = str(checkpoint_dir)
    usr_args["checkpoint_path"] = source.describe()
    usr_args["checkpoint_metadata"] = metadata
    return model


def eval(env, model: Pi0Model, obs):  # noqa: A001 - the evaluator's interface name
    if model.prompt is None:
        model.prompt = io.resolve_prompt("original", getattr(env, "_current_instruction", None))
    space = env.unwrapped.single_action_space
    low, high = np.asarray(space.low, np.float64).reshape(-1), np.asarray(space.high, np.float64).reshape(-1)
    rec = model.record
    env_seed = getattr(env.unwrapped, "_robosyn_episode_seed", None)
    if env_seed is None:
        raise Pi0Error("the evaluator recorded no episode seed (_robosyn_episode_seed); refusing unseeded noise")
    if rec["env_seed"] is None:
        rec["env_seed"] = int(env_seed)
    elif rec["env_seed"] != int(env_seed):
        raise Pi0Error(f"episode seed changed within the episode: {rec['env_seed']} -> {env_seed}")
    request = io.encode(obs, model.prompt)
    noise = call_noise(model.noise_base, rec["env_seed"], rec["inferences"], model.noise_shape)
    request[NOISE_KEY] = noise
    digest = noise_digest(noise)
    t0 = time.time()
    out = model.client.infer(request)
    dt = time.time() - t0
    if out.get("research_noise_digest") != digest:
        raise Pi0Error(f"the server used other noise: {out.get('research_noise_digest')} != {digest}")
    rec["noise_digests"].append(digest)
    model.process_calls += 1
    rec["includes_process_first_call"] |= model.process_calls == 1
    rec["inferences"] += 1
    rec["round_trip_s"].append(dt)
    rec["server_infer_ms"].append(float(out.get("policy_timing", {}).get("infer_ms", float("nan"))))
    actions, report = io.decode(out["actions"], low, high, PI0_STEP)
    rec["gripper_values"] += 2 * len(actions)
    rec["gripper_outside_unit"] += report["gripper_outside_unit"]
    rec["arm_outside_action_space"] += report["arm_outside_action_space"]
    if not report["finite"]:
        rec["nonfinite_outputs"] += 1
        raise FloatingPointError("the pi0 policy returned non-finite actions")
    a64 = np.asarray(out["actions"], np.float64)[:len(actions), :14]
    excursion = np.maximum(low[:14] - a64, a64 - high[:14])  # > 0 outside the joint limits; the engine clamps
    rec["arm_values"] += len(io.ARMS) * len(actions)
    for d in io.ARMS:
        if (excursion[:, d] > 0).any():
            rec["arm_excursion_max_by_dim"][d] = max(rec["arm_excursion_max_by_dim"].get(d, 0.0), float(excursion[:, d].max()))
        n_beyond = int((excursion[:, d] > 0.02).sum())
        if n_beyond:
            rec["arm_beyond_tol_by_dim"][d] = rec["arm_beyond_tol_by_dim"].get(d, 0) + n_beyond
    import torch  # the simulator's; imported here so the module loads without it (tests)

    device = env.unwrapped.device
    final_obs, info, truncated = obs, None, False
    for a in actions:
        final_obs, _, _, truncated, info = env.step(torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0))
        rec["actions_executed"] += 1
        if env.get_wrapper_attr("is_task_success")():
            break
        if bool(np.asarray(truncated.detach().cpu() if hasattr(truncated, "detach") else truncated).any()):
            break
    return final_obs, info, truncated, [dt]


def episode_metadata(model: Pi0Model) -> dict:
    r = model.record
    rt, si = list(r["round_trip_s"]), list(r["server_infer_ms"])
    cold = None
    if r["includes_process_first_call"] and rt:
        cold = {"round_trip_s": rt[0], "server_infer_ms": si[0]}
        rt, si = rt[1:], si[1:]
    return {"backend": "pi0", "prompt": model.prompt, "R": PI0_STEP, "inferences": r["inferences"],
            "noise_base": model.noise_base, "noise_derivation": NOISE_DERIVATION, "env_seed": r["env_seed"],
            "noise_digests": list(r["noise_digests"]), "actions_executed": r["actions_executed"],
            "cold_first_call": cold, "warm_round_trip_all_s": [round(x, 4) for x in rt],
            "gripper_values": r["gripper_values"], "gripper_outside_unit": r["gripper_outside_unit"],
            "arm_values": r["arm_values"], "arm_outside_action_space": r["arm_outside_action_space"],
            "arm_beyond_0.02_by_dim": {str(k): v for k, v in sorted(r["arm_beyond_tol_by_dim"].items())},
            "arm_excursion_max_by_dim": {str(k): v for k, v in sorted(r["arm_excursion_max_by_dim"].items())},
            "nonfinite_outputs": r["nonfinite_outputs"]}
