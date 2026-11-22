"""REMEMBER: cognitive map, working memory, episodic memory and semantic memory."""

import json
from collections import defaultdict
from pathlib import Path

from . import config as C
from .world import in_bounds, nbrs4


class CognitiveMap:
    """The robot's own internal map, built only from what it has sensed."""

    def __init__(self):
        self.known = {}  # cell -> "wall" | "free"
        self.smoke, self.survivors, self.fire = set(), set(), set()
        self.heat_pos = defaultdict(int)  # times heat was detected at a cell
        self.heat_neg = defaultdict(int)  # times heat was NOT detected at a cell
        self.visited = set()

    def update(self, p):
        for c, info in p.cells.items():
            self.known[c] = "wall" if info["wall"] else "free"
            (self.smoke.add if info["smoke"] else self.smoke.discard)(c)
            (self.survivors.add if info["survivor"] else self.survivors.discard)(c)
        for c, hot in p.heat.items():
            (self.heat_pos if hot else self.heat_neg)[c] += 1
        self.visited.add(p.pos)

    def is_wall(self, c):
        return self.known.get(c) == "wall"

    def frontier(self):
        """Known open cells that border unexplored space."""
        return [c for c, t in self.known.items() if t == "free" and any(n not in self.known for n in nbrs4(c))]

    def unknown_around(self, c, r=C.SENSOR_RADIUS):
        x, y = c
        return sum(
            1
            for dx in range(-r, r + 1)
            for dy in range(-r, r + 1)
            if in_bounds((x + dx, y + dy)) and (x + dx, y + dy) not in self.known
        )

    def coverage(self):
        return len(self.known) / (C.GRID_W * C.GRID_H)


class WorkingMemory:
    """Short-term, limited-capacity store (7 slots, after Miller's 7 +/- 2)."""

    CAPACITY = 7

    def __init__(self):
        self.slots = []

    def load(self, items):
        self.slots = list(items)[: self.CAPACITY]


class EpisodicMemory:
    """Time-stamped personal experiences of the current mission."""

    def __init__(self):
        self.events = []  # (step, cell, kind, text)

    def record(self, step, cell, kind, text):
        self.events.append((step, cell, kind, text))

    def recent(self, n=4):
        return self.events[-n:]

    def cells_of(self, kind):
        return {e[1] for e in self.events if e[2] == kind}


class SemanticMemory:
    """Long-term general facts learned across many missions; saved to JSON."""

    def __init__(self, path=None, learn=True):
        self.path = Path(path) if path else None
        self.learn = learn
        self.smoke_entered = 0
        self.fire_found = 0
        self.missions = 0
        self.total_rescued = 0
        if self.path and self.path.exists():
            self.__dict__.update(
                {
                    k: v
                    for k, v in json.loads(self.path.read_text()).items()
                    if k in ("smoke_entered", "fire_found", "missions", "total_rescued")
                }
            )

    def p_fire_given_smoke(self):
        a, b = C.PRIOR_FIRE_ALPHA, C.PRIOR_FIRE_BETA
        return (a + self.fire_found) / (a + b + self.smoke_entered)

    def observe_smoke(self, was_fire):
        if self.learn:
            self.smoke_entered += 1
            self.fire_found += int(was_fire)

    def end_mission(self, rescued):
        if self.learn:
            self.missions += 1
            self.total_rescued += rescued

    def facts(self):
        p = self.p_fire_given_smoke()
        return [
            f"Smoke w/o heat cue hides fire {p:.0%} of the time (n={self.smoke_entered})",
            f"Experience: {self.missions} missions, {self.total_rescued} survivors saved",
        ]

    def save(self, path=None):
        path = Path(path) if path else self.path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "smoke_entered": self.smoke_entered,
                        "fire_found": self.fire_found,
                        "missions": self.missions,
                        "total_rescued": self.total_rescued,
                        "p_fire_given_smoke": round(self.p_fire_given_smoke(), 4),
                    },
                    indent=2,
                )
            )
