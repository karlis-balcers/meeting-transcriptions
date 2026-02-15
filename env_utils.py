from __future__ import annotations

from typing import Optional


def env_str(env: dict[str, str], name: str, default: Optional[str] = None) -> Optional[str]:
    value = env.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_int(env: dict[str, str], name: str, default: int, warnings: list[str]) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.append(f"Invalid integer for {name}='{raw}'. Using default {default}.")
        return default


def env_float(env: dict[str, str], name: str, default: float, warnings: list[str]) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.append(f"Invalid number for {name}='{raw}'. Using default {default}.")
        return default


def env_bool(env: dict[str, str], name: str, default: bool, warnings: list[str]) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default

    value = raw.strip().lower()
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}

    if value in truthy:
        return True
    if value in falsy:
        return False

    warnings.append(f"Invalid boolean for {name}='{raw}'. Using default {default}.")
    return default
