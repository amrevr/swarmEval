"""Command-line interface.

    swarmeval demo [--trials N] [--out DIR]
    swarmeval run --swarm pkg.module:swarm --benchmark bench.json [--trials N] [--out DIR]
    swarmeval report runs/baseline.json [--trace N]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Callable, List, Optional

from . import __version__
from .evaluate import EvaluationResult, analyze_runs, evaluate
from .export.jsonl import write_events_jsonl
from .recorder import SimClock
from .report.text import render_trajectory
from .runner import RunSet
from .schema import Benchmark


def _load_obj(spec: str) -> Any:
    if ":" not in spec:
        raise SystemExit(f"expected package.module:attr (got {spec!r})")
    mod, attr = spec.split(":", 1)
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), attr)


def _load_benchmark(spec: str) -> Benchmark:
    if ":" in spec and not os.path.exists(spec):
        obj = _load_obj(spec)
        return obj() if callable(obj) and not isinstance(obj, Benchmark) else obj
    return Benchmark.load(spec)


def _clock_factory(name: Optional[str]) -> Optional[Callable[[], Any]]:
    if name == "sim":
        return SimClock
    if name in (None, "wall"):
        return None
    return _load_obj(name)


def _write_outputs(result: EvaluationResult, out: Optional[str]) -> None:
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    result.save(os.path.join(out, f"{result.label}.json"))
    write_events_jsonl(result.runs.trajectories, os.path.join(out, f"{result.label}.events.jsonl"))
    print(f"\nwrote {out}/{result.label}.json and {out}/{result.label}.events.jsonl")


def cmd_demo(args: argparse.Namespace) -> int:
    from examples.research_swarm import build_benchmark, make_swarm  # type: ignore
    result = evaluate(make_swarm, build_benchmark(), trials=args.trials, clock_factory=SimClock, progress=True)
    print(result.report())
    if args.trace:
        print()
        print(render_trajectory(result.runs.trajectories[0]))
    _write_outputs(result, args.out)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    swarm = _load_obj(args.swarm)
    bench = _load_benchmark(args.benchmark)
    result = evaluate(swarm, bench, trials=args.trials, seed=args.seed, clock_factory=_clock_factory(args.clock),
                      label=args.label, progress=True)
    print(result.report())
    _write_outputs(result, args.out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    with open(args.path) as f:
        d = json.load(f)
    runs = RunSet.from_dict(d["runs"] if "runs" in d else d)
    result = analyze_runs(runs)
    print(result.report())
    if args.trace is not None:
        print()
        print(render_trajectory(runs.trajectories[args.trace]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="swarmeval", description="Record and evaluate multi-agent systems.")
    p.add_argument("--version", action="version", version=f"swarmeval {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the built-in simulated research swarm")
    d.add_argument("--trials", type=int, default=3)
    d.add_argument("--out", default=None, help="directory for JSON/JSONL outputs")
    d.add_argument("--trace", action="store_true", help="also print the first trajectory")
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("run", help="run a swarm on a benchmark and print the report")
    r.add_argument("--swarm", required=True, help="package.module:swarm_or_factory_or_run_fn")
    r.add_argument("--benchmark", required=True, help="path to benchmark JSON/JSONL or package.module:builder")
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--clock", default=None, help="'wall' (default), 'sim', or package.module:factory")
    r.add_argument("--label", default="baseline")
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_run)

    rp = sub.add_parser("report", help="re-analyse a saved run set")
    rp.add_argument("path")
    rp.add_argument("--trace", type=int, default=None, metavar="N", help="also print trajectory N")
    rp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
