"""Load ego_state_plotter JSON config."""

from __future__ import annotations

import copy
import functools
import json
import os
from typing import Any

_CONFIG_FILENAME = 'plotter_params.json'


def default_plotter_params() -> dict[str, Any]:
    return {
        'plot_update_rate': 10.0,
        'error_history_size': 500,
        'enable_plotting': True,
    }


def _config_search_paths() -> list[str]:
    paths: list[str] = []
    override = os.environ.get('CONTROL_PLOTTER_PARAMS')
    if override:
        paths.append(os.path.expanduser(override.strip()))
    module_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
    paths.append(os.path.normpath(os.path.join(module_dir, '..', 'config', _CONFIG_FILENAME)))
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = get_package_prefix('control')
        install_root = os.path.dirname(prefix)
        ws = os.path.dirname(install_root)
        paths.append(os.path.join(ws, 'src', 'control', 'config', _CONFIG_FILENAME))
    except Exception:
        pass
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('control')
        paths.append(os.path.join(share, 'config', _CONFIG_FILENAME))
    except Exception:
        pass
    return paths


@functools.lru_cache(maxsize=1)
def load_plotter_params() -> dict[str, Any]:
    defaults = default_plotter_params()
    for path in _config_search_paths():
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError('root must be a JSON object')
            cfg = copy.deepcopy(defaults)
            cfg.update({k: v for k, v in raw.items() if k in cfg})
            return cfg
        except Exception:
            continue
    return copy.deepcopy(defaults)
