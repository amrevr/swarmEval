"""Layer 2 -- efficiency: tokens, cost and latency for one trajectory.

Phase 1 reports what was directly recorded: token usage, call counts, cost,
wall clock time and per-agent time.  Critical path and parallelism analysis
need the execution graph and are Phase 2 work.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from ..schema import EventType, Trajectory


def union_length(intervals: List[Tuple[float, float]]) -> float:
    """Total length covered by a set of possibly overlapping intervals."""
    ivs = sorted((a, b) for a, b in intervals if b > a)
    total = 0.0
    cur_a, cur_b = None, None
    for a, b in ivs:
        if cur_a is None:
            cur_a, cur_b = a, b
        elif a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    if cur_a is not None:
        total += cur_b - cur_a
    return total


@dataclass
class AgentEfficiency:
    agent_id: str
    role: Optional[str] = None
    spans: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    messages_sent: int = 0
    artifacts_produced: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    self_time_s: float = 0.0     # span time excluding nested child agents
    busy_time_s: float = 0.0     # time inside own llm/tool calls
    idle_time_s: float = 0.0     # self time not spent in calls (waiting)
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


@dataclass
class EfficiencyMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    messages: int = 0
    message_tokens: int = 0
    cost_usd: Optional[float] = None
    cost_complete: bool = True
    wall_clock_s: float = 0.0
    agent_time_s: float = 0.0     # sum of agent self time (aggregate work)
    llm_latency_s: float = 0.0
    tool_latency_s: float = 0.0
    waiting_time_s: float = 0.0
    errors: int = 0
    per_agent: Dict[str, AgentEfficiency] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["per_agent"] = {k: v.to_dict() for k, v in self.per_agent.items()}
        return d


def compute_efficiency(traj: Trajectory) -> EfficiencyMetrics:
    m = EfficiencyMetrics()
    per: Dict[str, AgentEfficiency] = {}

    def agent(aid: str) -> AgentEfficiency:
        if aid not in per:
            per[aid] = AgentEfficiency(aid)
        return per[aid]

    busy: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for ev in traj.events:
        if ev.event_type == EventType.LLM_CALL.value:
            m.llm_calls += 1
            m.input_tokens += ev.input_tokens
            m.output_tokens += ev.output_tokens
            m.llm_latency_s += ev.duration_s
            if ev.cost_usd is None:
                m.cost_complete = False
            else:
                m.cost_usd = (m.cost_usd or 0.0) + ev.cost_usd
            if ev.span_id:
                busy[ev.span_id].append((ev.timestamp, ev.end_timestamp))
            if ev.agent_id:
                a = agent(ev.agent_id)
                a.llm_calls += 1
                a.input_tokens += ev.input_tokens
                a.output_tokens += ev.output_tokens
                if ev.cost_usd is not None:
                    a.cost_usd = (a.cost_usd or 0.0) + ev.cost_usd
            if ev.status == "error":
                m.errors += 1
                if ev.agent_id:
                    agent(ev.agent_id).errors += 1
        elif ev.event_type == EventType.TOOL_CALL.value:
            m.tool_calls += 1
            m.tool_latency_s += ev.duration_s
            if ev.span_id:
                busy[ev.span_id].append((ev.timestamp, ev.end_timestamp))
            if ev.agent_id:
                agent(ev.agent_id).tool_calls += 1
            if ev.status == "error":
                m.errors += 1
                if ev.agent_id:
                    agent(ev.agent_id).errors += 1
        elif ev.event_type == EventType.MESSAGE.value:
            m.messages += 1
            m.message_tokens += ev.content_tokens
            if ev.sender:
                agent(ev.sender).messages_sent += 1
        elif ev.event_type == EventType.ARTIFACT.value and ev.agent_id:
            agent(ev.agent_id).artifacts_produced += 1
        elif ev.event_type == EventType.ERROR.value:
            m.errors += 1
    m.total_tokens = m.input_tokens + m.output_tokens
    if m.llm_calls == 0:
        m.cost_usd = 0.0

    # span-based timing
    children: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for s in traj.spans.values():
        if s.parent_span_id and s.end is not None:
            children[s.parent_span_id].append((s.start, s.end))
    for s in traj.spans.values():
        if s.end is None:
            continue
        a = agent(s.agent_id)
        a.spans += 1
        a.role = a.role or s.role
        child_time = union_length([(max(s.start, c0), min(s.end, c1)) for c0, c1 in children.get(s.span_id, [])])
        self_time = max(0.0, s.duration - child_time)
        busy_time = union_length(busy.get(s.span_id, []))
        a.self_time_s += self_time
        a.busy_time_s += busy_time
        a.idle_time_s += max(0.0, self_time - busy_time)
    m.per_agent = per
    m.agent_time_s = sum(a.self_time_s for a in per.values())
    m.waiting_time_s = sum(a.idle_time_s for a in per.values())
    m.wall_clock_s = traj.duration
    return m
