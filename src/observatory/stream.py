"""
src/observatory/stream.py — CreditOps v2 Observatory

Re-exports StreamWindow and WindowedStream from src.observatory.simulator.stream.
Maintains backward compatibility across all imports.
"""

from src.observatory.simulator.stream import StreamWindow, WindowedStream

__all__ = ["StreamWindow", "WindowedStream"]
