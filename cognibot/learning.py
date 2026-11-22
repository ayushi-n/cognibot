"""LEARN: tabular Q-learning over the high-level options (adaptive behaviour)."""

import json
from pathlib import Path

from . import config as C
from .decision import OPTIONS


class QLearner:
    def __init__(self, path=None, epsilon=0.0):
        self.path = Path(path) if path else None
        self.epsilon = epsilon
        self.q = {}
        self.updates = 0
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            self.q, self.updates = data.get("q", {}), data.get("updates", 0)

    @staticmethod
    def state(health, battery, base_cost, survivor_known, frontier_exists):
        h = 0 if health < 40 else (1 if health < 75 else 2)
        margin = battery - (base_cost or 0)
        b = 0 if margin < 20 else (1 if margin < 80 else 2)
        return f"h{h}_b{b}_s{int(survivor_known)}_f{int(frontier_exists)}"

    def qv(self, s, a):
        return self.q.get(s, {}).get(a, 0.0)

    def choose(self, s, opts, rng):
        avail = [a for a in OPTIONS if opts[a][1] is not None]
        if not avail:
            return "RETREAT"
        if rng.random() < self.epsilon:
            return rng.choice(avail)
        return max(avail, key=lambda a: opts[a][0] + C.Q_SCALE * self.qv(s, a))

    def update(self, s, a, r, s2, done):
        target = r if done else r + C.GAMMA * max(self.qv(s2, b) for b in OPTIONS)
        row = self.q.setdefault(s, {})
        row[a] = self.qv(s, a) + C.ALPHA * (target - self.qv(s, a))
        self.updates += 1

    def save(self, path=None):
        path = Path(path) if path else self.path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"q": self.q, "updates": self.updates}, indent=2))
