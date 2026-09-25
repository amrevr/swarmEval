"""Human-readable text reports (terminal friendly)."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..schema import EventType, Trajectory, preview
from ..stats import Summary


def bar(frac: float, width: int = 16) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def _money(x: Optional[float], complete: bool = True) -> str:
    if x is None:
        return "unknown"
    s = f"${x:.2f}" if x >= 0.1 else f"${x:.4f}"
    return s if complete else s + "*"


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _spread(s: Summary, unit: str = "", digits: int = 1) -> str:
    return s.fmt(unit=unit, digits=digits)


def box(title: str, rows: Sequence[Any], width: int = 58) -> str:
    longest = max([len(r) + 3 for r in rows if isinstance(r, str)] + [width])
    inner = max(width, longest) - 2
    out = ["╭" + "─" * inner + "╮", "│" + title.center(inner) + "│", "├" + "─" * inner + "┤"]
    for row in rows:
        if row is None:
            out.append("├" + "─" * inner + "┤")
            continue
        if isinstance(row, tuple):
            label, value = row[0], row[1]
            flag = row[2] if len(row) > 2 else ""
            line = f" {label:<28}{value:>{inner - 32}} {flag:<2}"
        else:
            line = f" {row}"
        out.append("│" + line[:inner].ljust(inner) + "│")
    out.append("╰" + "─" * inner + "╯")
    return "\n".join(out)


def render_text(result: Any, show_agents: bool = True, show_tasks: bool = True) -> str:
    perf, eff = result.performance, result.efficiency
    n_runs = len(result.runs)
    rows: List[Any] = [
        f"{result.label}  ·  {result.benchmark}  ·  {perf.n_tasks} tasks × {perf.trials} trials = {n_runs} runs",
        None,
        ("Task Success", f"{perf.n_successes}/{n_runs}  ({_pct(perf.success_rate)})"),
        ("Mean Score", f"{perf.mean_score:.3f}"),
    ]
    if perf.constraint_pass_rate is not None:
        rows.append(("Constraint Pass", _pct(perf.constraint_pass_rate)))
    if perf.run_errors:
        rows.append(("Runs Crashed", f"{perf.run_errors}/{n_runs}", "⚠"))
    rows += [
        None,
        ("Cost / Run", _money(eff.cost_usd.mean, eff.cost_complete)),
        ("Cost / Successful Task", _money(eff.cost_per_success, eff.cost_complete)),
        ("Total Cost", _money(eff.cost_total, eff.cost_complete)),
        ("Tokens / Run", f"{eff.tokens.mean:,.0f}  (in {eff.input_tokens.mean:,.0f} / out {eff.output_tokens.mean:,.0f})"),
        ("LLM Calls / Run", f"{eff.llm_calls.mean:.1f}"),
        ("Tool Calls / Run", f"{eff.tool_calls.mean:.1f}"),
        ("Messages / Run", f"{eff.messages.mean:.1f}"),
        None,
        ("Wall Clock / Run", _spread(eff.wall_clock_s, "s")),
        ("Agent Time / Run", f"{eff.agent_time_s.mean:.1f}s"),
        ("Waiting Time / Run", f"{eff.waiting_time_s.mean:.1f}s"),
    ]
    if eff.errors.total:
        rows.append(("Errors (all runs)", f"{eff.errors.total:.0f}", "⚠"))
    parts = [box("SWARM EVALUATION", rows)]
    if not eff.cost_complete:
        parts.append("  * cost incomplete: some models have no pricing entry")
    if perf.trials > 1:
        parts.append("  per-trial success: " + ", ".join(_pct(x) for x in perf.per_trial))

    if show_agents and eff.per_agent:
        parts.append("")
        parts.append("AGENTS (per run, mean)")
        parts.append(f"  {'agent':<16}{'role':<12}{'tokens':<24}{'llm':>5}{'tools':>6}{'msgs':>6}{'time':>8}{'cost':>10}")
        for aid, u in sorted(eff.per_agent.items(), key=lambda kv: -kv[1]["token_share"]):
            parts.append(f"  {aid:<16}{(u.get('role') or '-')[:11]:<12}{bar(u['token_share'])} {u['token_share']*100:>4.0f}%"
                         f"{u['llm_calls']:>5.1f}{u['tool_calls']:>6.1f}{u['messages_sent']:>6.1f}"
                         f"{u['self_time_s']:>7.1f}s{_money(u['cost_usd']):>10}")

    if show_tasks and perf.per_task:
        parts.append("")
        parts.append("TASKS")
        parts.append(f"  {'task':<16}{'success':>9}{'score':>8}{'cost':>10}{'latency':>9}")
        for tid, t in perf.per_task.items():
            parts.append(f"  {tid:<16}{_pct(t['success_rate']):>9}{t['score']:>8.2f}{_money(t['cost_usd']):>10}"
                         f"{t['latency_s']:>8.1f}s")
    parts.append("")
    parts.append(f"Evaluators: {', '.join(perf.evaluators) or 'none'}")
    return "\n".join(parts)


def render_trajectory(traj: Trajectory, max_events: int = 200) -> str:
    """Plain-text listing of one trajectory: what happened, in order."""
    t0 = traj.started_at or (traj.events[0].timestamp if traj.events else 0.0)
    ev_status = "✓" if traj.succeeded else "✗"
    head = (f"run {traj.run_id}  task {traj.task_id}  trial {traj.trial}  status {traj.status}  "
            f"eval {ev_status} {traj.score:.2f}  {traj.duration:.1f}s  {traj.total_tokens:,} tokens  "
            f"{_money(traj.cost_usd, traj.cost_is_complete)}")
    lines = [head, f"  {'t(s)':>7}  {'event':<12}{'agent':<16}detail"]
    depth = {None: 0}
    for ev in traj.events[:max_events]:
        if ev.event_type == EventType.AGENT_START.value:
            depth[ev.span_id] = depth.get(traj.spans[ev.span_id].parent_span_id if ev.span_id in traj.spans else None, 0) + 1
        boundary = ev.event_type in (EventType.AGENT_START.value, EventType.AGENT_END.value)
        indent = "  " * max(0, depth.get(ev.span_id, 0) - (1 if boundary else 0))
        et = ev.event_type
        if et == EventType.LLM_CALL.value:
            detail = f"{ev.model or '?'}  {ev.input_tokens}→{ev.output_tokens} tok  {ev.duration_s:.2f}s  {_money(ev.cost_usd)}"
        elif et == EventType.TOOL_CALL.value:
            detail = f"{ev.tool_name}({preview(ev.tool_args, 60)})  {ev.duration_s:.2f}s"
        elif et == EventType.MESSAGE.value:
            detail = f"{ev.sender} → {ev.receiver}  [{ev.kind}]  {ev.content_tokens} tok"
        elif et == EventType.ARTIFACT.value:
            detail = f"{ev.kind}  {ev.content_tokens} tok"
        elif et == EventType.ARTIFACT_USE.value:
            detail = f"uses {ev.kind}"
        elif et == EventType.AGENT_START.value:
            detail = f"role={ev.kind or '-'} model={ev.model or '-'}"
        elif et == EventType.AGENT_END.value:
            detail = f"{ev.status}  {ev.duration_s:.2f}s"
        elif et == EventType.WAIT.value:
            detail = f"{ev.kind}  {ev.duration_s:.2f}s"
        elif et == EventType.EVAL.value:
            detail = f"{ev.content}  score={ev.metadata.get('score')}"
        else:
            detail = ev.error or preview(ev.content, 80)
        flag = "" if ev.status == "success" else f" [{ev.status}]"
        lines.append(f"  {ev.timestamp - t0:>7.2f}  {et:<12}{indent}{(ev.agent_id or '-'):<16}{detail}{flag}")
    if len(traj.events) > max_events:
        lines.append(f"  … {len(traj.events) - max_events} more events")
    return "\n".join(lines)
