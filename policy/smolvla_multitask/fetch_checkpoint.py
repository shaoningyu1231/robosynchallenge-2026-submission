# ----------------------------------------------------------------------------
# Download one checkpoint snapshot; run with the SmolVLA worker's Python.
#
#   python fetch_checkpoint.py --repo-id org/name --revision <40-char commit>
#
# Prints one JSON line {"path": ..., "revision": ...} on stdout; progress goes to
# stderr. The request is anonymous (token=False) even when a token is
# configured, so a checkpoint that only downloads for a logged-in user fails
# here, in our own tests, instead of on the organisers' machine.
# HF_ENDPOINT, HF_HOME and HF_HUB_OFFLINE are honoured as usual.
# ----------------------------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(args.repo_id, revision=args.revision, token=False))
    # A snapshot requested by commit hash is stored under that hash.
    print(json.dumps({"path": str(path), "revision": path.name}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
