"""Shared data structures for Re-ID datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

SplitName = Literal["train", "query", "gallery"]


@dataclass(frozen=True)
class ReIDSample:
    path: Path
    pid: int
    camid: int
    metadata: Mapping[str, object] = field(default_factory=dict)
