"""The simulated research swarm now lives in ``swarmeval.examples.research_swarm``
(so it ships with the package for ``swarmeval demo``).  This module re-exports
it for scripts that import ``examples.research_swarm`` from a checkout."""
from swarmeval.examples.research_swarm import *  # noqa: F401,F403
from swarmeval.examples.research_swarm import ResearchSwarm, build_benchmark, make_swarm, sim_clock  # noqa: F401

if __name__ == "__main__":  # pragma: no cover
    from swarmeval import evaluate
    print(evaluate(make_swarm, build_benchmark(), trials=3, clock_factory=sim_clock).report())
