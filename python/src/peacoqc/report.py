"""CSV-based report writer for :func:`peacoqc.peaco_qc`.

Provides a single ``append_row`` helper that writes a row to a CSV file,
creating it with a header if it doesn't exist. The column order is chosen
to overlap with the R package's ``PeacoQC_report.txt`` so the same CSV
can be consumed by :func:`peacoqc.peaco_qc_heatmap`.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping

REPORT_COLUMNS = [
    "Filename",
    "Nr. Measurements before cleaning",
    "Nr. Measurements after cleaning",
    "% Full analysis",
    "Analysis by",
    "% IT analysis",
    "% MAD analysis",
    "% Consecutive cells",
    "MAD",
    "IT limit",
    "Consecutive bins",
    "Events per bin",
    "Increasing/Decreasing channel",
]


def append_row(path: str | os.PathLike[str], row: Mapping[str, object]) -> None:
    """Append a single row of QC results to ``path``.

    If the file does not exist it is created with a header row. Only the
    values for :data:`REPORT_COLUMNS` are written, in order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(REPORT_COLUMNS)
        writer.writerow([_fmt(row.get(col)) for col in REPORT_COLUMNS])


def _fmt(value: object) -> str:
    if value is None:
        return "Not_used"
    if isinstance(value, float):
        if value != value:  # NaN
            return "Not_used"
        return f"{value:.6g}"
    return str(value)
