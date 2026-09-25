# SwarmEval

> Evaluation and observability for multi-agent AI systems.

SwarmEval evaluates a multi-agent system not only on whether it produced the right answer, but on **how the agents worked together to produce it**. The foundation for that is trajectory observability: every agent invocation, LLM call, tool call, message and artifact is recorded as a structured event, and every metric is computed from those events.

This is **Phase 1** of the [roadmap](ROADMAP.md): observability and basic evaluation. Coordination analysis, causal ablation and optimization are later phases and are not in this release.

```
Phase 1  Trajectory recorder · task format · metrics · report · JSON export · agent interface   ← this release
Phase 2  Execution graph · utilization · communication · redundancy · critical path · swarm smells
Phase 3  Ablation · failure injection · counterfactuals · confidence intervals
Phase 4  Architecture recommendations · validated optimization
```

Pure Python 3.9+, no runtime dependencies. Optional Anthropic SDK adapter. The design charter is in [docs/DESIGN.md](docs/DESIGN.md); the full research write-up is in [research.md](research.md).

---

## Install

```bash
pip install -e ".[dev]"        # from this repo
pip install -e ".[anthropic]"  # optional: Anthropic SDK adapter
```

## 60-second demo

The repo ships a simulated research swarm (no API keys) with a coordinator, three researchers, a critic and a synthesizer:

```bash
swarmeval demo --trials 3 --trace --out runs/demo
```

```
╭──────────────────────────────────────────────────────────────╮
│                       SWARM EVALUATION                       │
├──────────────────────────────────────────────────────────────┤
│ baseline  ·  research_briefs  ·  8 tasks × 3 trials = 24 runs│
├──────────────────────────────────────────────────────────────┤
│ Task Success                                19/24  (79.2%)   │
│ Mean Score                                           0.931   │
│ Constraint Pass                                     100.0%   │
├──────────────────────────────────────────────────────────────┤
│ Cost / Run                                           $0.12   │
│ Cost / Successful Task                               $0.16   │
│ Tokens / Run                16,996  (in 13,916 / out 3,080)  │
│ LLM Calls / Run                                       11.0   │
│ Tool Calls / Run                                       6.0   │
├──────────────────────────────────────────────────────────────┤
│ Wall Clock / Run                                40.0s ±0.0   │
│ Agent Time / Run                                     54.6s   │
╰──────────────────────────────────────────────────────────────╯

AGENTS (per run, mean)
  agent           role        tokens                    llm tools  msgs    time      cost
  synthesizer     synthesizer █████████░░░░░░░   58%  1.0   0.0   0.0   13.4s   $0.0629
  coordinator     coordinator █████░░░░░░░░░░░   30%  3.0   0.0   5.0   11.6s   $0.0370
  ...
```

`--trace` prints one trajectory event by event (timestamps, models, tokens, cost, messages, artifacts). `--out` writes `baseline.json` (the run set plus metrics) and `baseline.events.jsonl` (one event per line).

---

## Instrumenting a swarm (the agent interface)

A swarm is anything with `run(task, ctx)`. Everything the swarm does goes through the `RunContext`, which is the recorder.

```python
from swarmeval import Swarm, Task, Benchmark, evaluate

class MySwarm(Swarm):
    def run(self, task, ctx):
        with ctx.agent("planner", role="coordinator", model="claude-opus-5") as planner:
            plan = planner.llm(call_model, task.input, prompt=task.input)   # LLMResult carries real usage
            planner.send("researcher", plan, kind="task_assignment")

        with ctx.agent("researcher", role="research", model="claude-sonnet-5") as r:
            assignment = r.receive()[0]
            hits = r.call_tool("web_search", search, query=assignment.content)
            notes = r.produce("notes", summarise(hits))
            r.send("writer", notes.content, kind="result", artifacts=[notes])

        with ctx.agent("writer", role="synthesizer") as w:
            msg = w.receive()[0]
            with w.llm_call(model="claude-opus-5", input_artifacts=msg.artifacts, prompt=msg.content) as call:
                answer = call_model(msg.content)               # any client
                call.input_tokens, call.output_tokens = answer.usage.input, answer.usage.output
                call.output = answer.text
            final = w.produce("answer", answer.text)
        ctx.set_output(answer.text, artifacts=[final])
        return answer.text

bench = Benchmark("qa", [Task("q1", "What is …?", expected=["keyword"],
                              evaluator={"type": "contains_all", "keywords": ["keyword"]})])
result = evaluate(MySwarm(), bench, trials=3)
print(result.report())
result.save("runs/qa.json")
```

| Call | Records |
|---|---|
| `ctx.agent(id, role, model)` | an agent invocation span (nest them for hierarchies; `ctx.parallel(...)` for threads) |
| `span.llm(fn, …)` / `span.llm_call(...)` | an LLM call with model, tokens, cost, latency, prompt, `input_artifacts` |
| `span.call_tool(name, fn, **args)` / `span.tool_call(...)` | a tool call with arguments, result, latency and status |
| `span.send(to, content, kind, artifacts)` / `span.receive()` | messages through a per-agent mailbox |
| `span.produce(name, content)` / `span.consume(artifact)` | artifacts and who used them |
| `span.wait(reason, seconds)` / `span.waiting(reason)` | explicit waiting |
| `span.note(text)` / `span.error(msg)` | annotations and errors |

Return an `LLMResult(text, input_tokens, output_tokens, model)` from the function you pass to `span.llm` to record real usage; otherwise tokens are estimated from text length and flagged as estimated. Unknown models get a cost of `unknown`, never a silent `$0`.

**Anthropic SDK**: `swarmeval.integrations.anthropic.instrument_client(client)` records every `messages.create` made inside an agent span with real usage; `anthropic_llm(client, model)` returns a prompt → `LLMResult` callable for `span.llm`.

`evaluate()` also accepts a bare `run(task, ctx)` function or a zero-argument factory. Use `clock_factory=SimClock` for simulated swarms that advance a virtual clock.

---

## Task format

A `Task` has `task_id`, `input`, optional `expected`, `success_criteria`, `constraints`, `reference`, `category`, `difficulty` and an `evaluator`. Evaluators are JSON specs so benchmarks are plain files:

```json
{"name": "qa", "tasks": [
  {"task_id": "q1", "input": "…", "evaluator": {"type": "contains_all", "keywords": ["a", "b"], "forbidden": ["c"]}},
  {"task_id": "q2", "input": "…", "expected": "42", "evaluator": {"type": "exact_match"}}
]}
```

Built-in evaluators: `exact_match`, `contains_all` / `contains_any` (with `forbidden`), `regex`, `numeric`, `tool_use` (checks the trajectory), `constraints`, `composite`, or any callable `fn(task, output, trajectory)`. `Benchmark.load(path)` accepts JSON or JSONL; `Benchmark.validate()` checks ids and fields.

## What gets measured

| Layer | Metrics |
|---|---|
| **Performance** | success rate, mean score, constraint pass rate, per-task and per-trial breakdown, crashed runs |
| **Efficiency** | input/output tokens, LLM and tool call counts, messages, cost (pricing table), cost per successful task, wall clock, aggregate agent time, waiting time, per-agent tokens/cost/calls/time |

Repeated trials use seeds derived from `(seed, task, trial)` so a run set is reproducible. Means and spreads are reported; confidence intervals are Phase 3.

## Outputs

* `EvaluationResult.report()` – text report (above); `.summary()` / `.to_dict()` / `.save(path)` – JSON including the full run set.
* `swarmeval.export.write_events_jsonl(trajectories, path)` – flat event stream for pandas / DuckDB.
* `RunSet.save/load`, `Trajectory.save/load` – trajectories are the source of truth; `swarmeval report runs.json` recomputes everything from them.
* `render_trajectory(traj)` – the event-by-event text trace.

## CLI

```bash
swarmeval demo [--trials N] [--out DIR] [--trace]
swarmeval run    --swarm pkg.mod:factory --benchmark bench.json --trials 3 --out runs [--clock sim]
swarmeval report runs/baseline.json [--trace N]
```

## Layout

```
swarmeval/
  schema.py        events, artifacts, spans, tasks, benchmarks, trajectories
  recorder.py      RunContext / AgentSpan: the agent interface and recorder
  runner.py        Swarm, SwarmRunner, RunSet, repeated trials, seeds
  evaluators.py    deterministic evaluators with JSON specs
  metrics/         performance, efficiency
  report/          text report and trajectory trace
  export/          JSONL events
  integrations/    Anthropic SDK adapter
  pricing.py       model pricing table
  stats.py         descriptive statistics
  cli.py
examples/research_swarm.py   simulated swarm + benchmark used by the demo and tests
tests/
```

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```
