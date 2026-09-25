"""SwarmEval -- evaluation and observability for multi-agent AI systems.

Phase 1 (this release): trajectory recording, standardized task format,
basic performance/efficiency metrics, text reports and JSON/JSONL export.

Quick start::

    from swarmeval import evaluate
    result = evaluate(my_swarm, benchmark, trials=3)
    print(result.report())
"""
from . import evaluators
from .evaluate import EfficiencySummary, EvaluationResult, analyze_runs, evaluate
from .recorder import AgentSpan, LLMCall, LLMResult, Message, RunContext, SimClock, ToolCall, WallClock
from .runner import FunctionSwarm, RunOptions, RunSet, Swarm, SwarmRunner
from .schema import Artifact, Benchmark, Evaluation, Event, EventType, Span, Task, Trajectory
from .metrics import EfficiencyMetrics, PerformanceMetrics, compute_efficiency, compute_performance
from .export import read_events_jsonl, write_events_jsonl
from .report import render_text, render_trajectory

__version__ = "0.1.0"

__all__ = [
    "evaluate", "analyze_runs", "EvaluationResult", "EfficiencySummary",
    "Task", "Benchmark", "Trajectory", "Event", "EventType", "Span", "Artifact", "Evaluation",
    "RunContext", "AgentSpan", "Message", "LLMResult", "LLMCall", "ToolCall", "SimClock", "WallClock",
    "Swarm", "FunctionSwarm", "SwarmRunner", "RunOptions", "RunSet",
    "PerformanceMetrics", "EfficiencyMetrics", "compute_performance", "compute_efficiency",
    "read_events_jsonl", "write_events_jsonl", "render_text", "render_trajectory",
    "evaluators", "__version__",
]
