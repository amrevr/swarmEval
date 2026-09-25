"""Core data model for SwarmEval (Phase 1: trajectory schema and task format).

Everything downstream (metrics, reports, exports and the later coordination
and causal layers) is computed from the structures defined here.  The guiding
rule from the design:

    If the execution cannot be represented as structured events,
    it cannot be reliably analyzed.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MESSAGE = "message"
    ARTIFACT = "artifact"          # an artifact was produced
    ARTIFACT_USE = "artifact_use"  # an artifact was consumed
    WAIT = "wait"
    ERROR = "error"
    EVAL = "eval"
    ANNOTATION = "annotation"


class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class MessageKind(str, Enum):
    """Coarse classification of messages.  Recorded now so that the
    communication analysis planned for Phase 2 has data to work with."""
    TASK_ASSIGNMENT = "task_assignment"
    COORDINATION = "coordination"
    INFORMATION = "information"
    RESULT = "result"
    REQUEST = "request"
    FEEDBACK = "feedback"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def to_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(obj)


def estimate_tokens(obj: Any) -> int:
    """Cheap token estimate (~4 characters per token) used when a real count
    is unavailable.  Always prefer real usage numbers when you have them."""
    text = to_text(obj)
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 4.0)))


def content_hash(obj: Any) -> str:
    return hashlib.sha1(to_text(obj).encode("utf-8")).hexdigest()


def preview(obj: Any, limit: int = 240) -> str:
    text = to_text(obj)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """A single structured execution event.

    The schema is intentionally flat so it serialises to one JSON object per
    line.  Not every field applies to every event type; unused fields keep
    their defaults.
    """
    event_id: str
    run_id: str
    task_id: str
    event_type: str
    timestamp: float                       # seconds (wall clock or simulated)
    agent_id: Optional[str] = None
    span_id: Optional[str] = None          # the agent invocation this belongs to
    parent_event_id: Optional[str] = None
    status: str = Status.SUCCESS.value
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    tool_name: Optional[str] = None
    tool_args: Any = None
    sender: Optional[str] = None
    receiver: Optional[str] = None
    kind: Optional[str] = None             # message kind / artifact name / wait reason / role
    content: Any = None                    # message content, output preview, ...
    content_tokens: int = 0
    content_hash: Optional[str] = None
    input_artifacts: List[str] = field(default_factory=list)
    output_artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def end_timestamp(self) -> float:
        return self.timestamp + (self.latency_ms or 0.0) / 1000.0

    @property
    def duration_s(self) -> float:
        return (self.latency_ms or 0.0) / 1000.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Artifact:
    """An intermediate or final product of an agent (notes, code, a decision)."""
    artifact_id: str
    name: str
    producer: Optional[str]
    span_id: Optional[str]
    produced_at: float
    content: Any
    content_hash: str
    tokens: int
    event_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Artifact":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Span:
    """One invocation of an agent (start..end)."""
    span_id: str
    agent_id: str
    role: Optional[str]
    parent_span_id: Optional[str]
    start: float
    end: Optional[float] = None
    status: str = Status.SUCCESS.value
    model: Optional[str] = None
    start_event_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Span":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# Tasks & benchmarks (the standardized task format)
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """One benchmark task.

    ``evaluator`` is either an ``Evaluator`` instance or a JSON spec such as
    ``{"type": "contains_all", "keywords": [...]}`` so benchmarks can live in
    plain files.
    """
    task_id: str
    input: Any
    category: str = "general"
    expected: Any = None
    success_criteria: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    reference: Any = None
    evaluator: Any = None
    difficulty: Optional[str] = None      # "simple" | "moderate" | "hard"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValueError("task_id must be a non-empty string")
        if self.input is None:
            raise ValueError(f"task {self.task_id!r} has no input")
        if self.difficulty not in (None, "simple", "moderate", "hard"):
            raise ValueError(f"task {self.task_id!r}: difficulty must be simple|moderate|hard")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        ev = self.evaluator
        if ev is not None and not isinstance(ev, dict):
            spec = getattr(ev, "to_spec", None)
            d["evaluator"] = spec() if callable(spec) else {"type": type(ev).__name__}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Benchmark:
    name: str
    tasks: List[Task]
    default_evaluator: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def get(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(task_id)

    def validate(self) -> None:
        seen = set()
        for t in self.tasks:
            t.validate()
            if t.task_id in seen:
                raise ValueError(f"duplicate task_id {t.task_id!r} in benchmark {self.name!r}")
            seen.add(t.task_id)

    def to_dict(self) -> Dict[str, Any]:
        ev = self.default_evaluator
        if ev is not None and not isinstance(ev, dict):
            spec = getattr(ev, "to_spec", None)
            ev = spec() if callable(spec) else None
        return {
            "name": self.name,
            "tasks": [t.to_dict() for t in self.tasks],
            "default_evaluator": ev,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Benchmark":
        return cls(
            name=d.get("name", "benchmark"),
            tasks=[Task.from_dict(t) for t in d.get("tasks", [])],
            default_evaluator=d.get("default_evaluator"),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Benchmark":
        with open(path) as f:
            if path.endswith(".jsonl"):
                tasks = [Task.from_dict(json.loads(line)) for line in f if line.strip()]
                return cls(name=path, tasks=tasks)
            return cls.from_dict(json.load(f))

    def subset(self, task_ids: Iterable[str]) -> "Benchmark":
        ids = set(task_ids)
        return Benchmark(self.name, [t for t in self.tasks if t.task_id in ids],
                         self.default_evaluator, dict(self.metadata))


# --------------------------------------------------------------------------- #
# Evaluation results
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    success: bool
    score: float
    evaluator: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)
    constraints_passed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evaluation":
        return cls(
            success=bool(d.get("success", False)),
            score=float(d.get("score", 0.0)),
            evaluator=d.get("evaluator", "unknown"),
            details=d.get("details", {}),
            constraints_passed=d.get("constraints_passed"),
        )


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    """The complete structured record of one run of a swarm on one task."""
    run_id: str
    task_id: str
    task: Task
    seed: int = 0
    trial: int = 0
    events: List[Event] = field(default_factory=list)
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    spans: Dict[str, Span] = field(default_factory=dict)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    final_output: Any = None
    output_artifacts: List[str] = field(default_factory=list)
    status: str = Status.SUCCESS.value
    error: Optional[str] = None
    evaluation: Optional[Evaluation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors ------------------------------------------------
    @property
    def duration(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return max(0.0, self.ended_at - self.started_at)

    def events_of(self, *types: str) -> List[Event]:
        wanted = {t.value if isinstance(t, EventType) else t for t in types}
        return [e for e in self.events if e.event_type in wanted]

    def llm_calls(self) -> List[Event]:
        return self.events_of(EventType.LLM_CALL)

    def tool_calls(self) -> List[Event]:
        return self.events_of(EventType.TOOL_CALL)

    def messages(self) -> List[Event]:
        return self.events_of(EventType.MESSAGE)

    def agent_ids(self) -> List[str]:
        seen: List[str] = []
        for s in sorted(self.spans.values(), key=lambda s: s.start):
            if s.agent_id not in seen:
                seen.append(s.agent_id)
        return seen

    def spans_for(self, agent_id: str) -> List[Span]:
        return sorted((s for s in self.spans.values() if s.agent_id == agent_id),
                      key=lambda s: s.start)

    @property
    def input_tokens(self) -> int:
        return sum(e.input_tokens for e in self.llm_calls())

    @property
    def output_tokens(self) -> int:
        return sum(e.output_tokens for e in self.llm_calls())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> Optional[float]:
        costs = [e.cost_usd for e in self.llm_calls()]
        known = [c for c in costs if c is not None]
        if not known:
            return None
        return float(sum(known))

    @property
    def cost_is_complete(self) -> bool:
        return all(e.cost_usd is not None for e in self.llm_calls())

    @property
    def succeeded(self) -> bool:
        return bool(self.evaluation and self.evaluation.success)

    @property
    def score(self) -> float:
        return float(self.evaluation.score) if self.evaluation else 0.0

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task": self.task.to_dict(),
            "seed": self.seed,
            "trial": self.trial,
            "events": [e.to_dict() for e in self.events],
            "artifacts": {k: a.to_dict() for k, a in self.artifacts.items()},
            "spans": {k: s.to_dict() for k, s in self.spans.items()},
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "final_output": self.final_output,
            "output_artifacts": self.output_artifacts,
            "status": self.status,
            "error": self.error,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        return cls(
            run_id=d["run_id"],
            task_id=d["task_id"],
            task=Task.from_dict(d["task"]),
            seed=d.get("seed", 0),
            trial=d.get("trial", 0),
            events=[Event.from_dict(e) for e in d.get("events", [])],
            artifacts={k: Artifact.from_dict(a) for k, a in d.get("artifacts", {}).items()},
            spans={k: Span.from_dict(s) for k, s in d.get("spans", {}).items()},
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            final_output=d.get("final_output"),
            output_artifacts=d.get("output_artifacts", []),
            status=d.get("status", "success"),
            error=d.get("error"),
            evaluation=Evaluation.from_dict(d["evaluation"]) if d.get("evaluation") else None,
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        with open(path) as f:
            return cls.from_dict(json.load(f))
