from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def discover_runs(root: str | Path = "outputs") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])
