"""Regression tests for rohitp30's review nits on PR #4 (and the fan-in/out
note on PR #3, whose second definition lives here).  Recording message body
tokens at the source is held for rohitp30's answer on PR #4."""
from swarmeval import (EvaluationResult, LexicalSimilarity, RunContext, SimClock, Task, build_graph,
                       compute_communication, compute_redundancy, evaluate)
from swarmeval.examples.research_swarm import build_benchmark, make_swarm


def make_ctx():
    ctx = RunContext(Task("t", "q"), clock=SimClock(), seed=0)
    ctx.start()
    return ctx


# -- fan-in / fan-out: the graph and communication agree ---------------------- #
def test_graph_and_communication_fan_agree_on_invocation_only_edge():
    ctx = make_ctx()
    with ctx.agent("a") as a:
        with a.sub_agent("b"):                              # a invokes b; no message, no artifact
            pass
    traj = ctx.finish()
    g = build_graph(traj)
    cm = compute_communication(traj, graph=g)
    assert g.fan_out("a") == cm.per_agent["a"].fan_out
    assert g.fan_in("b") == cm.per_agent["b"].fan_in


# -- "hash only" only once the text is really gone ----------------------------- #
def test_in_memory_run_compares_real_text_even_with_capture_off():
    res = evaluate(make_swarm, build_benchmark(1), clock_factory=SimClock, capture_text=False)
    assert not compute_redundancy(res.runs.trajectories[0]).degraded


def test_saved_capture_off_run_stays_hash_only_and_save_does_not_mutate(tmp_path):
    """Guard (passes before and after the fix): the fix must not flag the
    in-memory run on save, and a saved run must still read as hash-only."""
    res = evaluate(make_swarm, build_benchmark(1), clock_factory=SimClock, capture_text=False)
    before = dict(res.runs.trajectories[0].metadata)
    res.save(str(tmp_path / "r.json"))
    assert res.runs.trajectories[0].metadata == before
    loaded = EvaluationResult.load(str(tmp_path / "r.json")).runs.trajectories[0]
    assert compute_redundancy(loaded).degraded


# -- lexical candidates are capped (reviewer: optional) ----------------------- #
def test_lexical_candidates_capped_when_one_phrase_is_everywhere():
    sim = LexicalSimilarity(max_pairs=500)
    pairs = sim.candidates([f"no reliable information found item {i}" for i in range(200)])
    assert len(pairs) <= 500 and sim.truncated
