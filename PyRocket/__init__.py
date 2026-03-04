"""PyRocket package entrypoints."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import SimulationContext


def run_simulation(*args, **kwargs):
    from .app import run_simulation as _run_simulation

    return _run_simulation(*args, **kwargs)


def build_runtime_context(*args, **kwargs):
    from .runtime import build_runtime_context as _build_runtime_context

    return _build_runtime_context(*args, **kwargs)


def __getattr__(name):
    if name == "SimulationContext":
        from .runtime import SimulationContext

        return SimulationContext
    raise AttributeError(f"module 'PyRocket' has no attribute '{name}'")


__all__ = ["SimulationContext", "build_runtime_context", "run_simulation"]
