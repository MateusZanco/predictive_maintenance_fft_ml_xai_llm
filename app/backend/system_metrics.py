from __future__ import annotations

from pathlib import Path
from typing import Any

import psutil


def _read_temperature_from_psutil() -> tuple[float | None, str | None]:
    try:
        temperatures = psutil.sensors_temperatures(fahrenheit=False)
    except (AttributeError, NotImplementedError, OSError):
        return None, None

    if not temperatures:
        return None, None

    preferred_keys = ("cpu_thermal", "soc_thermal", "coretemp")
    for key in preferred_keys:
        if key in temperatures and temperatures[key]:
            entry = temperatures[key][0]
            return float(entry.current), key

    for key, entries in temperatures.items():
        if entries:
            return float(entries[0].current), key

    return None, None


def _read_temperature_from_sysfs() -> tuple[float | None, str | None]:
    thermal_root = Path("/sys/class/thermal")
    if not thermal_root.exists():
        return None, None

    for zone_dir in sorted(thermal_root.glob("thermal_zone*")):
        temp_file = zone_dir / "temp"
        type_file = zone_dir / "type"
        if not temp_file.exists():
            continue

        try:
            raw_value = temp_file.read_text(encoding="utf-8").strip()
            temperature = float(raw_value)
        except (OSError, ValueError):
            continue

        if temperature > 1000:
            temperature /= 1000.0

        source = zone_dir.name
        if type_file.exists():
            try:
                source = type_file.read_text(encoding="utf-8").strip() or source
            except OSError:
                pass

        return temperature, source

    return None, None


def read_system_metrics() -> dict[str, Any]:
    per_core_percent = [float(value) for value in psutil.cpu_percent(interval=0.1, percpu=True)]
    cpu_percent = float(sum(per_core_percent) / len(per_core_percent)) if per_core_percent else 0.0
    memory = psutil.virtual_memory()
    cpu_busy_threshold_percent = 5.0

    temperature_c, temperature_source = _read_temperature_from_psutil()
    if temperature_c is None:
        temperature_c, temperature_source = _read_temperature_from_sysfs()

    return {
        "cpu_percent": cpu_percent,
        "cpu_logical_cores": int(psutil.cpu_count(logical=True) or 0),
        "cpu_physical_cores": int(psutil.cpu_count(logical=False) or 0),
        "cpu_busy_cores": int(sum(1 for value in per_core_percent if value >= cpu_busy_threshold_percent)),
        "cpu_busy_threshold_percent": cpu_busy_threshold_percent,
        "cpu_per_core_percent": per_core_percent,
        "memory_percent": float(memory.percent),
        "memory_used_mb": float(memory.used) / (1024 * 1024),
        "memory_total_mb": float(memory.total) / (1024 * 1024),
        "temperature_c": temperature_c,
        "temperature_source": temperature_source,
    }
