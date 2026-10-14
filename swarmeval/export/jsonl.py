"""One-event-per-line export: easy to load into pandas, DuckDB or a warehouse."""
from __future__ import annotations

import json
from typing import Any, Iterable, List, Union

from ..schema import Event, Trajectory


def write_events_jsonl(source: Union[Trajectory, Iterable[Trajectory]], path: str) -> int:
    trajs = [source] if isinstance(source, Trajectory) else list(source)
    n = 0
    with open(path, "w") as f:
        for t in trajs:
            for ev in t.events:
                f.write(json.dumps(ev.to_dict(), default=str) + "\n")
                n += 1
    return n


def read_events_jsonl(path: str) -> List[Event]:
    out: List[Event] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(Event.from_dict(json.loads(line)))
    return out
