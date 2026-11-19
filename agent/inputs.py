"""
User inputs besides the goal: loading a dataset file and a data dictionary.

The data dictionary is free text (or a ``{column: description}`` mapping) that
tells every LLM stage what columns and coded values mean, e.g. that taxi
``payment_type`` 1 = credit card and 2 = cash. It flows into the plan,
code-gen, repair, critique and summary prompts.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from agent.dates import looks_like_date_name, parses_as_dates

# Upper bound on dictionary text embedded in every prompt.
MAX_DICTIONARY_CHARS = 8000


def format_data_dictionary(dictionary: str | Mapping[str, Any] | None) -> str:
    """
    Normalise a data dictionary to prompt text.

    Accepts free text, or a mapping of column name to a description (a string,
    or any JSON-serialisable value such as a ``{code: label}`` dict). Returns
    ``""`` for ``None`` / empty input. Text longer than
    :data:`MAX_DICTIONARY_CHARS` is truncated with a marker.
    """
    if dictionary is None:
        return ""
    if isinstance(dictionary, Mapping):
        lines = []
        for column, description in dictionary.items():
            if not isinstance(description, str):
                description = json.dumps(description, ensure_ascii=False, default=str)
            lines.append(f"- {column}: {description}")
        text = "\n".join(lines)
    else:
        text = str(dictionary)
    text = text.strip()
    if len(text) > MAX_DICTIONARY_CHARS:
        text = text[:MAX_DICTIONARY_CHARS].rstrip() + "\n… (data dictionary truncated)"
    return text


def load_data_dictionary(path: str | Path) -> str:
    """
    Read a data dictionary file and return prompt text.

    ``.json`` files must hold an object (column -> description) or a string;
    any other file (``.md``, ``.txt``, ``.yaml`` ...) is used as plain text.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if not isinstance(data, (dict, str)):
            raise ValueError(
                f"{path}: a JSON data dictionary must be an object or a string"
            )
        return format_data_dictionary(data)
    return format_data_dictionary(raw)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV or Parquet file into a DataFrame.

    String columns with a date-like name whose values parse as dates are
    converted to datetime64, so generated code and the date detection see real
    timestamps. Parquet needs ``pyarrow`` (``uv sync --extra parquet``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise ImportError(
                "Reading Parquet needs pyarrow: run `uv sync --extra parquet` "
                "or `pip install pyarrow`."
            ) from exc
    elif suffix in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", low_memory=False)
    else:
        raise ValueError(f"Unsupported data file type {suffix!r}: use CSV or Parquet")

    for col in df.columns:
        if looks_like_date_name(str(col)) and parses_as_dates(df[col]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
