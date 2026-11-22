"""The CogniBot cognitive agent (Perceive -> Remember -> Reason -> Plan -> Decide -> Act -> Learn)
and two baseline agents used for comparison."""

import random
import time

from . import config as C
from .decision import evaluate_options, u_explore, u_rescue
from .learning import QLearner
from .memory import CognitiveMap, EpisodicMemory, WorkingMemory
from .perception import sense
from .planning import astar, dijkstra
from .reasoning import KnowledgeBase
from .world import manhattan, nbrs4


class CogniBot:
    name = "CogniBot"

    def __init__(self, semantic, qlearner=None, learn=True, seed=0):
        self.sem, self.q, self.learn = semantic, qlearner, learn
        self.kb = KnowledgeBase()
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.cmap, self.wm, self.epi = CognitiveMap(), WorkingMemory(), EpisodicMemory()
        self.prev = None  # (state, action, reward) waiting for its Q-update
        self.intention = None  # (action, target) the robot is committed to
        self.trace = {}
        self.decision_ms = []

    def _score(self, a, u, s):
        return u + (C.Q_SCALE * self.q.qv(s, a) if self.q is not None else 0.0)

    def _commit(self, action, target, opts, dist, s, pos):
        """Intention commitment: keep the current goal unless it is finished, invalid,
        or a new option is clearly better. Prevents dithering between goals."""
        it = self.intention
        if it is not None:
            ia, it_t = it
            valid = False
            if ia == "RESCUE" and it_t in self.cmap.survivors and it_t in dist and opts["RESCUE"][1] is not None:
                valid, u = True, u_rescue(dist[it_t])
            elif (
                ia == "EXPLORE"
                and it_t != pos
                and it_t in dist
                and self.cmap.unknown_around(it_t) > 0
                and opts["EXPLORE"][1] is not None
            ):
                valid, u = True, u_explore(self.cmap.unknown_around(it_t), dist[it_t])
            elif ia == "RETREAT" and opts["RETREAT"][1] is not None:
                valid, u = True, opts["RETREAT"][0]
            if valid and action != ia:
                if self._score(action, opts[action][0], s) <= C.COMMIT_MARGIN * self._score(ia, u, s) + 0.3:
                    action, target = ia, it_t
            elif valid and action == ia and ia != "RETREAT":
                target = it_t
        self.intention = (action, target)
        return action, target

    # ------------------------------------------------------------------ one cognitive cycle
    def step(self, m):
        t0 = time.perf_counter()
        # 1. PERCEIVE
        p = sense(m)
        new_surv = [s for s in p.survivors if s not in self.cmap.survivors]
        self.cmap.update(p)
        for s in new_surv:
            self.epi.record(m.steps, s, "seen", f"t{m.steps}: saw survivor at {s}")
        for c, hot in p.heat.items():
            if hot:
                self.epi.record(m.steps, c, "heat", f"t{m.steps}: felt heat at {c}")

        # 2-3. REMEMBER + REASON
        danger, fired = self.kb.infer_danger(self.cmap, self.epi, self.sem)
        dist = dijkstra(m.pos, self.cmap, danger)
        flags, fired2 = self.kb.infer_goals(
            m.health, m.battery, dist.get(C.BASE), self.cmap, len(m.rescued), m.total_survivors
        )
        fired += fired2

        # 4. DECIDE (utility + learned Q-value)
        opts = evaluate_options(m.pos, self.cmap, dist, flags, m.health, m.battery, m.steps)
        s = QLearner.state(
            m.health, m.battery, dist.get(C.BASE), flags["survivor_known"], opts["EXPLORE"][1] is not None
        )
        if self.q is not None and self.learn and self.prev:
            ps, pa, pr = self.prev
            self.q.update(ps, pa, pr, s, False)
        if self.q is not None:
            action = self.q.choose(s, opts, self.rng)
        else:
            avail = [a for a in opts if opts[a][1] is not None] or ["RETREAT"]
            action = max(avail, key=lambda a: opts[a][0])
        target = opts[action][1]
        action, target = self._commit(action, target, opts, dist, s, m.pos)
        # 5. PLAN (A*)
        path = astar(m.pos, target, self.cmap, danger) if target is not None else None
        self.decision_ms.append((time.perf_counter() - t0) * 1000)

        # 6. ACT
        if action == "RETREAT" and m.pos == C.BASE:
            events = m.finish_at_base()
            nxt = m.pos
        else:
            nxt = path[1] if path and len(path) > 1 else m.pos
            events = m.move(nxt)

        # 7. LEARN from the outcome
        r = C.R_STEP
        for e in events:
            if e == "rescued":
                r += C.R_RESCUE
                self.epi.record(m.steps, nxt, "rescued", f"t{m.steps}: RESCUED survivor at {nxt}")
            elif e == "burned":
                r += C.R_FIRE
                self.cmap.fire.add(nxt)
                self.epi.record(m.steps, nxt, "burned", f"t{m.steps}: BURNED by hidden fire at {nxt}")
            elif e == "home":
                r += C.R_HOME
            elif e == "destroyed":
                r += C.R_DESTROYED
            elif e == "battery_dead":
                r += C.R_BATTERY_DEAD
        if "smoke_entered" in events and self.kb.cell_rule.get(nxt) == "R3":
            self.sem.observe_smoke("burned" in events)  # learn only from uninformed smoke cells
        if self.q is not None and self.learn:
            if m.done:
                self.q.update(s, action, r, None, True)
                self.prev = None
            else:
                self.prev = (s, action, r)
        if m.done:
            self.sem.end_mission(len(m.rescued))

        self.wm.load(
            [
                f"Position {m.pos}",
                f"Health {m.health}",
                f"Battery {m.battery}",
                f"Survivors in view: {len(p.survivors)}",
                f"Smoke in view: {len(p.smoke)}",
                f"Heat cues: {sum(p.heat.values())}",
                f"Goal: {action}",
            ]
        )
        self.trace = dict(
            percept=p,
            danger=danger,
            fired=fired,
            opts=opts,
            state=s,
            action=action,
            target=target,
            path=path or [m.pos],
            events=events,
            reward=r,
            qvals={a: (self.q.qv(s, a) if self.q else 0.0) for a in opts},
        )
        return self.trace


class RandomAgent:
    """Baseline 1: random walk, no memory, no reasoning."""

    name = "Random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def reset(self):
        self.trace = {}
        self.decision_ms = []

    def step(self, m):
        t0 = time.perf_counter()
        p = sense(m)
        moves = [n for n in nbrs4(m.pos) if not p.cells[n]["wall"]] or [m.pos]
        nxt = self.rng.choice(moves)
        self.decision_ms.append((time.perf_counter() - t0) * 1000)
        m.move(nxt)
        self.trace = {"path": [m.pos], "action": "RANDOM"}
        return self.trace


class ReactiveAgent:
    """Baseline 2: purely reactive (stimulus -> response), no memory or map.
    Moves toward a survivor it can currently see and avoids cells that feel hot right now."""

    name = "Reactive"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def reset(self):
        self.trace = {}
        self.decision_ms = []

    def step(self, m):
        t0 = time.perf_counter()
        p = sense(m)
        moves = [n for n in nbrs4(m.pos) if not p.cells[n]["wall"] and not p.heat.get(n, False)] or [m.pos]
        if p.survivors:
            goal = min(p.survivors, key=lambda s: manhattan(s, m.pos))
            nxt = min(moves, key=lambda n: manhattan(n, goal))
        else:
            nxt = self.rng.choice(moves)
        self.decision_ms.append((time.perf_counter() - t0) * 1000)
        m.move(nxt)
        self.trace = {"path": [m.pos], "action": "REACT"}
        return self.trace
