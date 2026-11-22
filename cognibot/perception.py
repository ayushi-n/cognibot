"""PERCEIVE: simulated camera (5x5 view) and a noisy short-range heat sensor."""

from dataclasses import dataclass, field

from . import config as C
from .world import in_bounds, nbrs4


@dataclass
class Percept:
    pos: tuple
    cells: dict = field(default_factory=dict)  # cell -> {"wall","smoke","survivor"}
    heat: dict = field(default_factory=dict)  # adjacent smoky cell -> heat detected?

    @property
    def survivors(self):
        return [c for c, i in self.cells.items() if i["survivor"]]

    @property
    def smoke(self):
        return [c for c, i in self.cells.items() if i["smoke"]]


def sense(mission):
    """The robot cannot see fire directly: fire hides inside smoke.
    It only gets a noisy heat reading from the 4 cells next to it."""
    w, (x, y), R = mission.world, mission.pos, C.SENSOR_RADIUS
    p = Percept(pos=mission.pos)
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            c = (x + dx, y + dy)
            if in_bounds(c):
                p.cells[c] = {"wall": c in w.walls, "smoke": c in w.smoke, "survivor": c in w.survivors}
    for n in nbrs4(mission.pos):
        if n in w.smoke:
            prob = C.HEAT_DETECT_PROB if n in w.fires else C.HEAT_FALSE_ALARM
            p.heat[n] = mission.rng.random() < prob
    return p
