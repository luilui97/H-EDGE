"""Source Bridge (Section V-B): fact ports and source-mention windows.

The component ranks by cue overlap + identity bonus + 0.05/(1+dense rank).
Port-role metadata is intentionally excluded; acceptance checks roles later.
"""
from components.source_qa.bridge import BridgeSelector as SourceBridge

__all__ = ["SourceBridge"]
