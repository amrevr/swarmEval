"""Layer 1 -- task performance aggregated over a set of runs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..schema import Trajectory
from ..stats import mean


@dataclass
class PerformanceMetrics:
    n_runs: int = 0
    n_tasks: int = 0
    trials: int = 0
    n_successes: int = 0
    success_rate: float = 0.0
    mean_score: float = 0.0
    constraint_pass_rate: Optional[float] = None
    run_errors: int = 0
    per_task: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_trial: List[float] = field(default_factory=list)
    evaluators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_runs": self.n_runs, "n_tasks": self.n_tasks, "trials": self.trials,
            "n_successes": self.n_successes, "success_rate": self.success_rate,
            "mean_score": self.mean_score, "constraint_pass_rate": self.constraint_pass_rate,
            "run_errors": self.run_errors, "per_task": self.per_task, "per_trial": self.per_trial,
            "evaluators": self.evaluators,
        }


def compute_performance(trajectories: Sequence[Trajectory]) -> PerformanceMetrics:
    pm = PerformanceMetrics()
    pm.n_runs = len(trajectories)
    if not trajectories:
        return pm
    pm.n_successes = sum(1 for t in trajectories if t.succeeded)
    pm.success_rate = pm.n_successes / pm.n_runs
    pm.mean_score = mean([t.score for t in trajectories])
    cons = [t.evaluation.constraints_passed for t in trajectories
            if t.evaluation and t.evaluation.constraints_passed is not None]
    if cons:
        pm.constraint_pass_rate = sum(1 for c in cons if c) / len(cons)
    pm.run_errors = sum(1 for t in trajectories if t.status != "success")
    by_task: Dict[str, List[Trajectory]] = defaultdict(list)
    by_trial: Dict[int, List[bool]] = defaultdict(list)
    for t in trajectories:
        by_task[t.task_id].append(t)
        by_trial[t.trial].append(t.succeeded)
    pm.n_tasks = len(by_task)
    pm.trials = max(by_trial) + 1 if by_trial else 0
    for tid, ts in by_task.items():
        pm.per_task[tid] = {"success_rate": mean([1.0 if x.succeeded else 0.0 for x in ts]),
                            "score": mean([x.score for x in ts]), "n": len(ts),
                            "cost_usd": mean([x.cost_usd or 0.0 for x in ts]),
                            "latency_s": mean([x.duration for x in ts]),
                            "category": ts[0].task.category, "difficulty": ts[0].task.difficulty}
    pm.per_trial = [mean([1.0 if s else 0.0 for s in by_trial[i]]) for i in sorted(by_trial)]
    pm.evaluators = sorted({t.evaluation.evaluator for t in trajectories if t.evaluation})
    return pm
