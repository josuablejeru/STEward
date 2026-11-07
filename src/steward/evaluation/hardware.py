"""Hardware discovery and GPU memory sampling. Works without a GPU."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass

from steward.evaluation.records import Hardware


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """Basic data for GPU 0, from ``nvidia-smi``."""

    name: str
    memory_total_mib: int
    driver: str


def _nvidia_smi(query: str) -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def gpu_info() -> GpuInfo | None:
    """Return GPU 0 information, or None when no NVIDIA GPU is visible."""
    out = _nvidia_smi("name,memory.total,driver_version")
    if out is None:
        return None
    try:
        name, total, driver = (part.strip() for part in out.splitlines()[0].split(","))
        return GpuInfo(name=name, memory_total_mib=int(float(total)), driver=driver)
    except ValueError:
        return None


def gpu_memory_used_mib() -> int | None:
    """Return current used memory on GPU 0 (all processes), or None."""
    out = _nvidia_smi("memory.used")
    try:
        return int(float(out.splitlines()[0])) if out else None
    except ValueError:
        return None


def describe_hardware(provider_version: str | None = None) -> Hardware:
    """Return the hardware description stored on records."""
    gpu = gpu_info()
    return Hardware(
        platform=platform.platform(terse=True),
        cpu_count=os.cpu_count(),
        gpu_name=gpu.name if gpu else None,
        gpu_memory_total_mib=gpu.memory_total_mib if gpu else None,
        gpu_driver=gpu.driver if gpu else None,
        provider_version=provider_version,
        note=None if gpu else "No NVIDIA GPU visible to nvidia-smi.",
    )


class PeakMemorySampler:
    """Sample GPU 0 memory in a background thread and keep the peak.

    The value includes every process on the GPU, not only the model.
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        """Create a sampler. Call :meth:`start` and :meth:`stop`."""
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            used = gpu_memory_used_mib()
            if used is None:
                return
            self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            self._stop.wait(self.interval_s)

    def start(self) -> PeakMemorySampler:
        """Start sampling (does nothing when no GPU is visible)."""
        if shutil.which("nvidia-smi") is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> int | None:
        """Stop sampling and return the peak in MiB, or None."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        return self.peak_mib
