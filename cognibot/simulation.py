"""Headless episode runner shared by evaluation and tests."""

from statistics import mean

from .world import Mission, World


def run_episode(agent, seed):
    world = World(seed)
    m = Mission(world, seed)
    agent.reset()
    while not m.done:
        agent.step(m)
    cov = agent.cmap.coverage() if hasattr(agent, "cmap") else None
    return {
        "agent": agent.name,
        "seed": seed,
        "rescued": len(m.rescued),
        "total": m.total_survivors,
        "steps": m.steps,
        "fire_hits": m.fire_hits,
        "health": max(m.health, 0),
        "outcome": m.outcome,
        "recovered": int(m.recovered),
        "score": round(m.score(), 3),
        "coverage": round(cov, 3) if cov is not None else "",
        "decision_ms": round(mean(agent.decision_ms), 4) if agent.decision_ms else 0,
    }
