"""Unit and integration tests (test cases T01-T10 in the report).
Run with:  python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognibot import config as C
from cognibot.agent import CogniBot
from cognibot.decision import evaluate_options
from cognibot.learning import QLearner
from cognibot.memory import CognitiveMap, EpisodicMemory, SemanticMemory
from cognibot.perception import sense
from cognibot.planning import astar
from cognibot.reasoning import KnowledgeBase
from cognibot.simulation import run_episode
from cognibot.world import Mission, World


def open_map():
    cm = CognitiveMap()
    for x in range(C.GRID_W):
        for y in range(C.GRID_H):
            cm.known[(x, y)] = "free"
    return cm


class TestCogniBot(unittest.TestCase):
    def test_T01_astar_shortest_path(self):
        path = astar((0, 0), (5, 3), open_map(), {})
        self.assertEqual(len(path) - 1, 8)  # Manhattan distance on open grid

    def test_T02_astar_walls(self):
        cm = open_map()
        for y in range(C.GRID_H):
            cm.known[(3, y)] = "wall"  # full wall splits the map
        self.assertIsNone(astar((0, 0), (5, 0), cm, {}))
        cm.known[(3, 7)] = "free"  # open one gap
        path = astar((0, 0), (5, 0), cm, {})
        self.assertIn((3, 7), path)

    def test_T03_rule_smoke_and_heat_means_fire(self):
        cm, kb = open_map(), KnowledgeBase()
        cm.smoke.add((4, 4))
        cm.heat_pos[(4, 4)] = 1
        danger, fired = kb.infer_danger(cm, EpisodicMemory(), SemanticMemory(learn=False))
        self.assertAlmostEqual(danger[(4, 4)], 0.9)
        self.assertTrue(any(f.startswith("R1") for f in fired))

    def test_T04_planner_avoids_dangerous_smoke(self):
        cm = open_map()
        danger = {(2, y): 0.9 for y in range(0, 4)}  # dangerous strip on the direct route
        path = astar((0, 0), (4, 0), cm, danger)
        self.assertFalse(any(c in danger for c in path))

    def test_T05_semantic_memory_learns(self):
        sm = SemanticMemory(learn=True)
        before = sm.p_fire_given_smoke()
        for i in range(20):
            sm.observe_smoke(was_fire=(i % 4 == 0))  # 25% of smoky cells were fire
        self.assertGreater(sm.p_fire_given_smoke(), before)

    def test_T06_perception_radius(self):
        m = Mission(World(5), 5)
        p = sense(m)
        self.assertEqual(max(max(abs(c[0] - m.pos[0]), abs(c[1] - m.pos[1])) for c in p.cells), C.SENSOR_RADIUS)

    def test_T07_fire_damage(self):
        w = World(3)
        m = Mission(w, 3)
        f = next(iter(w.fires))
        m.pos = (f[0] - 1, f[1]) if f[0] > 0 else (f[0] + 1, f[1])
        if m.pos in w.walls:
            w.walls.discard(m.pos)
        ev = m.move(f)
        self.assertIn("burned", ev)
        self.assertEqual(m.health, C.MAX_HEALTH - C.FIRE_DAMAGE)

    def test_T08_low_battery_forces_retreat(self):
        cm = open_map()
        cm.survivors.add((10, 10))
        dist = {(10, 10): 18, C.BASE: 30, (5, 5): 5}
        flags = {"low_health": False, "low_battery": True, "survivor_known": True, "all_rescued": False}
        opts = evaluate_options((6, 6), cm, dist, flags, 100, 35, 200)
        best = max((a for a in opts if opts[a][1] is not None), key=lambda a: opts[a][0])
        self.assertEqual(best, "RETREAT")

    def test_T09_qlearning_update_and_persistence(self):
        q = QLearner()
        q.update("s", "RESCUE", 10.0, None, True)
        self.assertAlmostEqual(q.qv("s", "RESCUE"), C.ALPHA * 10.0)
        with tempfile.TemporaryDirectory() as d:
            q.save(os.path.join(d, "q.json"))
            self.assertAlmostEqual(QLearner(os.path.join(d, "q.json")).qv("s", "RESCUE"), C.ALPHA * 10.0)

    def test_T10_full_mission_terminates(self):
        r = run_episode(CogniBot(SemanticMemory(learn=False), None, learn=False), 11)
        self.assertLessEqual(r["steps"], C.MAX_BATTERY)
        self.assertIn(r["outcome"], ("Returned to base", "Robot destroyed", "Battery depleted"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
