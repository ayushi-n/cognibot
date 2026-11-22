"""Ground-truth disaster world and the robot's mission state."""

import random
from collections import deque

from . import config as C


def in_bounds(c):
    return 0 <= c[0] < C.GRID_W and 0 <= c[1] < C.GRID_H


def nbrs4(c):
    x, y = c
    return [n for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if in_bounds(n)]


def nbrs8(c):
    x, y = c
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (dx or dy) and in_bounds((x + dx, y + dy)):
                out.append((x + dx, y + dy))
    return out


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class World:
    """The true environment. The robot never reads this directly, only through sensors."""

    def __init__(self, seed):
        self.seed = seed
        rng = random.Random(seed)
        while not self._generate(rng):
            pass

    def _generate(self, rng):
        W, H = C.GRID_W, C.GRID_H
        walls, fires = set(), set()
        for _ in range(C.N_DEBRIS):
            x, y = rng.randrange(W), rng.randrange(H)
            horiz, length = rng.random() < 0.5, rng.randint(2, 5)
            for i in range(length):
                c = (x + i, y) if horiz else (x, y + i)
                if in_bounds(c) and chebyshev(c, C.BASE) > 2:
                    walls.add(c)
        free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in walls]
        for _ in range(C.N_FIRE_SEEDS):
            cand = [f for f in free if chebyshev(f, C.BASE) > 4 and f not in fires]
            c = rng.choice(cand)
            fires.add(c)
            if rng.random() < 0.6:
                opts = [n for n in nbrs4(c) if n not in walls and chebyshev(n, C.BASE) > 4]
                if opts:
                    fires.add(rng.choice(opts))
        smoke = set(fires)
        for f in fires:
            smoke.update(n for n in nbrs8(f) if n not in walls)
        # survivors must be reachable from base (walls block, fire does not)
        seen, q = {C.BASE}, deque([C.BASE])
        while q:
            c = q.popleft()
            for n in nbrs4(c):
                if n not in walls and n not in seen:
                    seen.add(n)
                    q.append(n)
        cand = sorted(c for c in seen if c not in smoke and chebyshev(c, C.BASE) > 3)
        if len(cand) < C.N_SURVIVORS or len(seen) < 0.6 * W * H:
            return False
        self.walls, self.fires, self.smoke = walls, fires, smoke
        self.survivors = set(rng.sample(cand, C.N_SURVIVORS))
        self.initial_survivors = set(self.survivors)
        return True


class Mission:
    """Robot body state inside one episode (position, health, battery, outcome)."""

    def __init__(self, world, seed=None):
        self.world = world
        s = world.seed if seed is None else seed
        self.rng = random.Random(s * 7919 + 1)  # sensor noise
        self.pos = C.BASE
        self.health = C.MAX_HEALTH
        self.battery = C.MAX_BATTERY
        self.steps = 0
        self.rescued = []
        self.fire_hits = 0
        self.done = False
        self.outcome = ""
        self.total_survivors = len(world.survivors)
        self.trail = [C.BASE]

    def move(self, cell):
        """Move to an adjacent cell (or stay). Returns a list of event strings."""
        w = self.world
        if cell != self.pos and (not in_bounds(cell) or cell in w.walls or manhattan(cell, self.pos) != 1):
            cell = self.pos  # illegal move = stay in place
        events = []
        self.pos = cell
        self.steps += 1
        self.battery -= 1
        self.trail.append(cell)
        if cell in w.survivors:
            w.survivors.remove(cell)
            self.rescued.append(cell)
            events.append("rescued")
        if cell in w.smoke:
            events.append("smoke_entered")
        if cell in w.fires:
            self.health -= C.FIRE_DAMAGE
            self.fire_hits += 1
            events.append("burned")
        if self.health <= 0:
            self.done, self.outcome = True, "Robot destroyed"
            events.append("destroyed")
        elif self.battery <= 0:
            self.done, self.outcome = True, "Battery depleted"
            events.append("battery_dead")
        return events

    def finish_at_base(self):
        if self.pos == C.BASE and not self.done:
            self.done, self.outcome = True, "Returned to base"
            return ["home"]
        return []

    @property
    def recovered(self):
        return self.outcome == "Returned to base"

    def score(self):
        """Same scoring formula for every agent (matches the RL reward)."""
        r = len(self.rescued) * C.R_RESCUE + self.fire_hits * C.R_FIRE + self.steps * C.R_STEP
        if self.outcome == "Returned to base":
            r += C.R_HOME
        elif self.outcome == "Robot destroyed":
            r += C.R_DESTROYED
        elif self.outcome == "Battery depleted":
            r += C.R_BATTERY_DEAD
        return r
