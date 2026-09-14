# ----------------------------------------------------------------------------
# Per-episode seed for SmolVLA's flow-matching noise
# ----------------------------------------------------------------------------
#
# Without an explicit noise tensor LeRobot samples fresh Gaussian noise for every
# action chunk from the global RNG, so the same checkpoint on the same scene can
# succeed once and fail once (it did: acceptance run, 2026-09-13). With
# smolvla_noise_seed set, every episode gets its own generator, seeded from that
# base and the episode's environment seed, and advanced once per model forward.
# Two runs that accept the same environment seeds then feed the policy the same
# noise, and two models compared on those seeds see the same noise too.
#
# Standard library only, so the rule can be tested without torch.

from __future__ import annotations

import hashlib

DERIVATION = "int.from_bytes(sha256(f'smolvla-noise:{base}:{env_seed}').digest()[:8], 'big') & (2**63 - 1)"
PER_FORWARD = (
    "torch CPU Generator seeded with the episode's noise seed; one "
    "torch.randn((1, chunk_size, max_action_dim), float32) draw per model forward"
)


def parse_base(value) -> int | None:
    """smolvla_noise_seed from the config: None (off) or a non-negative integer."""
    if value in (None, "", "None", "null", "none"):
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())   # a command-line override arrives as a string
    raise ValueError(f"smolvla_noise_seed must be null or a non-negative integer, got {value!r}")


def policy_noise_seed(base: int, env_seed: int) -> int:
    """The noise seed of one episode; a torch generator accepts any value below 2**63."""
    digest = hashlib.sha256(f"smolvla-noise:{int(base)}:{int(env_seed)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)
