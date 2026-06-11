"""Load control package JSON configs with lru_cache (one parse per process)."""

from __future__ import annotations

import copy
import functools
import json
import os
from typing import Any

_CONFIG_FILENAME = 'carla_control_params.json'


def default_control_params() -> dict[str, Any]:
    return {
        'vehicle': {
            'wheelbase': 2.7,
            'max_steer_angle': 1.22,
            'wheel_count': 4,
        },
        'control_loop': {
            'control_dt': 0.05,
            'max_speed': 6.0,
            'min_speed': 0.5,
            'max_acceleration': 2.0,
            'max_deceleration': -3.0,
            'dead_zone_throttle': 0.1,
            'beta_zero_speed_thresh_mps': 0.15,
            'switch_threshold': 1.0,
            'goal_arrival_distance': 0.5,
            'waypoint_interval': 0.5,
            'reference_time_horizon': 5.0,
            'reference_path_publish_every_n_cycles': 2,
        },
        'filter': {
            'alpha_acc': 0.1,
            'alpha_spd': 0.1,
            'alpha_beta': 0.1,
            'alpha_throttle': 0.1,
            'alpha_brake': 0.6,
            'alpha_x': 0.3,
            'alpha_y': 0.3,
            'alpha_yaw': 0.3,
            'initial_throttle': 0.3,
        },
        'ekf': {
            'P0_diag': [1.0, 1.0, 0.5],
            'Q_diag': [0.1, 0.1, 0.01],
            'R_diag': [0.5, 0.5, 0.05],
            'dt': 0.13,
        },
        'noise': {
            'enable': True,
            'odom_std_x': 0.025,
            'odom_std_y': 0.01,
            'odom_std_yaw': 0.01,
            'status_std_velocity': 0.1,
            'imu_std_accel': 0.05,
        },
        'lqr': {
            'q_x': 100.0,
            'q_y': 100.0,
            'q_theta': 10.0,
            'r_v': 1.0,
            'r_delta': 50.0,
            'max_steer': 1.22,
            'lr_ratio': 0.5,
            'v_r_floor': 0.15,
            'curvature_dt': 0.05,
        },
        'speed_controller': {
            'kp': 2.0,
            'ki': 0.15,
            'kd': 0.05,
            'output_limits': [-2.0, 2.0],
            'integral_limit': 1.0,
        },
        'throttle_controller': {
            'kp': 0.3,
            'ki': 0.15,
            'kd': 0.0,
            'output_limits': [0.0, 1.0],
            'integral_limit': 10.0,
        },
        'brake_controller': {
            'kp': 0.15,
            'ki': 0.1,
            'kd': 0.0,
            'output_limits': [0.0, 1.0],
            'integral_limit': 5.0,
        },
        'stanley': {
            'k': 3.5,
            'epsilon': 0.3,
            'max_steer': 1.22,
            'filter_alpha': 0.2,
            'lookahead_base': 1.5,
            'lookahead_gain': 0.5,
            'curvature_feedforward_gain': 0.0,
        },
    }


def _colcon_workspace_src_config_path() -> str | None:
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = get_package_prefix('control')
        install_root = os.path.dirname(prefix)
        ws = os.path.dirname(install_root)
        return os.path.join(ws, 'src', 'control', 'config', _CONFIG_FILENAME)
    except Exception:
        return None


def _config_search_paths() -> list[str]:
    paths: list[str] = []
    override = os.environ.get('CONTROL_PARAMS')
    if override:
        paths.append(os.path.expanduser(override.strip()))
    module_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
    paths.append(os.path.normpath(os.path.join(module_dir, '..', 'config', _CONFIG_FILENAME)))
    ws_src = _colcon_workspace_src_config_path()
    if ws_src:
        paths.append(os.path.normpath(ws_src))
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('control')
        paths.append(os.path.join(share, 'config', _CONFIG_FILENAME))
    except Exception:
        pass
    return paths


def _merge_params(defaults: dict, raw: dict) -> dict:
    cfg = copy.deepcopy(defaults)
    for key, val in raw.items():
        if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key].update(val)
        elif key in cfg:
            cfg[key] = val
    return cfg


@functools.lru_cache(maxsize=1)
def load_control_params() -> dict[str, Any]:
    defaults = default_control_params()
    for path in _config_search_paths():
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError('root must be a JSON object')
            return _merge_params(defaults, raw)
        except Exception:
            continue
    return copy.deepcopy(defaults)


def control_section(name: str) -> dict[str, Any]:
    return load_control_params()[name]


def pid_from_section(section: dict, dt: float) -> dict:
    lim = section['output_limits']
    if isinstance(lim, (list, tuple)):
        lim = (float(lim[0]), float(lim[1]))
    il = section.get('integral_limit')
    return {
        'kp': float(section['kp']),
        'ki': float(section['ki']),
        'kd': float(section['kd']),
        'dt': dt,
        'output_limits': lim,
        'integral_limit': None if il is None else float(il),
    }
