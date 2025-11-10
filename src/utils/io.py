from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_pickle(path: str | Path) -> Any:
    return joblib.load(path)


def write_json(data: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as fp:
        json.dump(data, fp, indent=indent)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r") as fp:
        return json.load(fp)


def save_dataframe(df: pd.DataFrame, path: str | Path, format: str = "parquet", index: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if format == "parquet":
        df.to_parquet(path, index=index)
    elif format == "csv":
        df.to_csv(path, index=index)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file suffix: {path.suffix}")


def save_artifact(obj: Any, path: str | Path) -> None:
    if isinstance(obj, pd.DataFrame):
        save_dataframe(obj, path)
    else:
        save_pickle(obj, path)


def list_files(directory: str | Path, pattern: Optional[str] = None) -> Iterable[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    if pattern is None:
        yield from directory.iterdir()
    else:
        yield from directory.glob(pattern)
