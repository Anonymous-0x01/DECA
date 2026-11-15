"""Cache the winning config of an operator tune.

Each run is identified by the GPU name and the operator configuration (type,
shape, dtype, and the few flags that change the kernel). The winner is stored
as JSON under `.cache/<gpu>/<operator>/<identity>.json`. A later run with the
same identity loads that config and skips the search.

The file records the config only. Timings are measured again on the next run.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass

# Deca repo root: this file lives at src/utils/tune_cache.py.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_ROOT = os.path.join(ROOT, ".cache")

# Readable filename order. Unknown keys are appended alphabetically.
_KEY_ORDER = (
    "M",
    "N",
    "K",
    "B",
    "H",
    "S_q",
    "S_kv",
    "D",
    "dtype",
    "acc",
    "acc_splits",
    "causal",
    "eps",
    "use_tc",
)


def _slug(text) -> str:
    """Filesystem-safe directory name. Spaces become underscores; path characters are dropped."""
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch in " .":
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "unknown"


def _ordered_keys(identity: dict) -> list:
    known = [key for key in _KEY_ORDER if key in identity]
    rest = sorted(key for key in identity if key not in _KEY_ORDER)
    return known + rest


def _token(key, value) -> str:
    """One identity field in the cache filename."""
    if isinstance(value, bool):
        return f"{key}{int(value)}"
    if isinstance(value, int):
        return f"{key}{value}"
    if isinstance(value, float):
        return f"{key}{format(value, '.6g')}"
    # dtype is written bare (`float16`) so the filename stays short.
    if key == "dtype":
        return _slug(value)
    return f"{key}-{_slug(value)}"


def _to_jsonable(config):
    """Dataclasses become dicts; tuples become lists so the file is plain JSON."""
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, (list, tuple)):
        return [_to_jsonable(item) for item in config]
    return config


def cache_path(gpu: str, op: str, identity: dict) -> str:
    """Path of the cache file for this GPU and operator configuration."""
    filename = "_".join(_token(key, identity[key]) for key in _ordered_keys(identity))
    return os.path.join(CACHE_ROOT, _slug(gpu), _slug(op), filename + ".json")


def load_tune_cache(gpu: str, op: str, identity: dict, config_cls=None):
    """Return the cached config, or None when there is no usable file.

    `config_cls` rebuilds a dataclass (`GemmConfig`, `FAConfig`, `RowConfig`).
    Pass None for a plain JSON value such as a tile-pattern list.
    """
    path = cache_path(gpu, op, identity)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        # Reject a file whose stored identity does not match the requested run.
        if payload.get("identity") != json.loads(json.dumps(identity)):
            return None
        raw = payload["config"]
        if config_cls is None:
            return raw
        return config_cls(**raw)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_tune_cache(gpu: str, op: str, identity: dict, config) -> str:
    """Write the winning config. A temp file is replaced into place so a crash cannot leave partial JSON."""
    path = cache_path(gpu, op, identity)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"identity": identity, "config": _to_jsonable(config)}
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def drop_tune_cache(gpu: str, op: str, identity: dict) -> None:
    """Remove a cache entry that failed to build or failed its correctness check."""
    path = cache_path(gpu, op, identity)
    if os.path.isfile(path):
        os.remove(path)
