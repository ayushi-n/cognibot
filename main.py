"""CogniBot live demo (pygame).

Run:  python main.py            -> uses the trained brain in memory/ if it exists
      python main.py --fresh    -> starts with an empty brain (watch it learn; trained brain is kept)
      python main.py --seed 7   -> a specific map

Keys: SPACE pause/resume | RIGHT/N one step (when paused) | UP/DOWN speed
      G god-view (show hidden fires) | R new map | L learning on/off
      S save brain | H heat-map of danger | ESC quit
"""

import argparse
import os
import random
from pathlib import Path

import pygame

from cognibot import config as C
from cognibot.agent import CogniBot
from cognibot.learning import QLearner
from cognibot.memory import SemanticMemory
from cognibot.reasoning import RULES
from cognibot.world import Mission, World

ROOT = Path(__file__).resolve().parent
MEM = ROOT / "memory"

CELL = 32
GX, GY = 16, 58  # grid origin
GW, GH = C.GRID_W * CELL, C.GRID_H * CELL
PX = GX + GW + 16  # panel x
WIN_W, WIN_H = PX + 500, 680

BG = (17, 20, 28)
PANEL = (26, 30, 42)
FOG = (34, 37, 48)
FREE = (62, 68, 86)
VISITED = (78, 86, 108)
WALL = (128, 96, 64)
SMOKE = (150, 152, 162)
FIRE = (240, 90, 40)
SURV = (64, 210, 130)
BASE_C = (80, 140, 255)
ROBOT = (255, 210, 60)
PATH = (90, 220, 255)
TXT = (226, 230, 238)
DIM = (140, 148, 165)
RED = (255, 95, 85)
GREEN = (90, 210, 120)
YEL = (255, 205, 80)
STAGE_COL = {
    "PERCEIVE": (102, 178, 255),
    "REMEMBER": (190, 140, 255),
    "REASON": (255, 160, 90),
    "PLAN": (90, 220, 255),
    "DECIDE": (255, 205, 80),
    "ACT": (110, 220, 140),
}
SPEEDS = [1, 2, 4, 8, 15, 30, 60]


def font(size, bold=False):
    return pygame.font.SysFont("consolas,dejavusansmono,couriernew,monospace", size, bold=bold)


class App:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption("CogniBot - Cognitive Rescue Robot (23RA4702 Mini Project)")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.f_title, self.f_head, self.f, self.f_small = font(20, True), font(14, True), font(13), font(12)
        self.clock = pygame.time.Clock()
        # --fresh uses a separate empty "brain" so the trained one in memory/ is never lost
        prefix = "fresh_" if args.fresh else ""
        self.sem_path, self.q_path = MEM / f"{prefix}semantic_memory.json", MEM / f"{prefix}q_table.json"
        if args.fresh:
            for f in (self.sem_path, self.q_path):
                f.unlink(missing_ok=True)
        self.sem = SemanticMemory(self.sem_path, learn=True)
        self.q = QLearner(self.q_path, epsilon=0.0)
        self.bot = CogniBot(self.sem, self.q, learn=True, seed=1)
        self.learning = True
        self.seed = args.seed if args.seed is not None else random.randint(1, 99999)
        self.speed_i, self.paused, self.god, self.heat = 3, False, args.god, True
        self.episode, self.end_timer = 0, 0
        self.new_mission(self.seed)

    def new_mission(self, seed):
        self.seed = seed
        self.world = World(seed)
        self.m = Mission(self.world, seed)
        self.bot.reset()
        self.trace = {}
        self.episode += 1
        self.end_timer = 0

    def save(self):
        self.sem.save(self.sem_path)
        self.q.save(self.q_path)

    def tick(self):
        if not self.m.done:
            self.trace = self.bot.step(self.m)
            if self.m.done and self.learning:
                self.save()

    # ------------------------------------------------------------------ drawing
    def cell_rect(self, c):
        return pygame.Rect(GX + c[0] * CELL, GY + c[1] * CELL, CELL, CELL)

    def text(self, s, pos, col=TXT, f=None):
        self.screen.blit((f or self.f).render(s, True, col), pos)

    def draw_grid(self):
        cm, w = self.bot.cmap, self.world
        danger = self.trace.get("danger", {})
        overlay = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        for x in range(C.GRID_W):
            for y in range(C.GRID_H):
                c, r = (x, y), self.cell_rect((x, y))
                known = c in cm.known
                if not known and not self.god:
                    pygame.draw.rect(self.screen, FOG, r)
                    continue
                if (c in w.walls) if self.god else cm.is_wall(c):
                    pygame.draw.rect(self.screen, WALL, r)
                    pygame.draw.line(self.screen, (95, 70, 45), r.topleft, r.bottomright, 2)
                    continue
                pygame.draw.rect(self.screen, VISITED if c in cm.visited else FREE, r)
                if not known:  # god view of unexplored cell
                    overlay.fill((0, 0, 0, 110))
                    self.screen.blit(overlay, r)
                smoky = (c in w.smoke) if self.god and not known else c in cm.smoke
                if smoky:
                    overlay.fill((*SMOKE, 120))
                    self.screen.blit(overlay, r)
                    for i in range(3):
                        pygame.draw.circle(
                            self.screen, (190, 192, 200), (r.x + 8 + i * 8, r.y + 12 + (i % 2) * 7), 4, 1
                        )
                if self.heat and c in danger and danger[c] > 0:
                    overlay.fill((255, 40, 30, int(40 + 150 * min(1, danger[c]))))
                    self.screen.blit(overlay, r)
                if c in cm.fire or (self.god and c in w.fires):
                    pygame.draw.circle(self.screen, FIRE, r.center, 9)
                    pygame.draw.circle(self.screen, YEL, r.center, 4)
        for x in range(C.GRID_W + 1):
            pygame.draw.line(self.screen, BG, (GX + x * CELL, GY), (GX + x * CELL, GY + GH))
        for y in range(C.GRID_H + 1):
            pygame.draw.line(self.screen, BG, (GX, GY + y * CELL), (GX + GW, GY + y * CELL))
        # base
        br = self.cell_rect(C.BASE).inflate(-6, -6)
        pygame.draw.rect(self.screen, BASE_C, br, border_radius=4)
        self.text("B", (br.x + 7, br.y + 3), (255, 255, 255), self.f_head)
        # survivors (known, or all in god view) and rescued ticks
        surv = cm.survivors | (w.survivors if self.god else set())
        for s in surv:
            r = self.cell_rect(s)
            pygame.draw.circle(self.screen, SURV, (r.centerx, r.y + 10), 5)
            pygame.draw.line(self.screen, SURV, (r.centerx, r.y + 15), (r.centerx, r.y + 24), 3)
            pygame.draw.line(self.screen, SURV, (r.centerx - 6, r.y + 18), (r.centerx + 6, r.y + 18), 2)
        for s in self.m.rescued:
            r = self.cell_rect(s)
            pygame.draw.lines(
                self.screen, GREEN, False, [(r.x + 8, r.y + 17), (r.x + 14, r.y + 23), (r.x + 25, r.y + 9)], 3
            )
        # sensor range
        sr = pygame.Rect(0, 0, (2 * C.SENSOR_RADIUS + 1) * CELL, (2 * C.SENSOR_RADIUS + 1) * CELL)
        sr.center = self.cell_rect(self.m.pos).center
        pygame.draw.rect(self.screen, (102, 178, 255), sr.clip(pygame.Rect(GX, GY, GW, GH)), 1)
        # planned path
        path = self.trace.get("path", [])
        if len(path) > 1:
            pygame.draw.lines(self.screen, PATH, False, [self.cell_rect(c).center for c in path], 3)
            pygame.draw.circle(self.screen, PATH, self.cell_rect(path[-1]).center, 6, 2)
        # robot
        rc = self.cell_rect(self.m.pos).center
        pygame.draw.circle(self.screen, ROBOT, rc, 11)
        pygame.draw.circle(self.screen, BG, rc, 11, 2)
        pygame.draw.circle(self.screen, BG, (rc[0] - 4, rc[1] - 2), 2)
        pygame.draw.circle(self.screen, BG, (rc[0] + 4, rc[1] - 2), 2)

    def bar(self, x, y, w, frac, col, label):
        pygame.draw.rect(self.screen, (50, 55, 70), (x, y, w, 12), border_radius=3)
        pygame.draw.rect(self.screen, col, (x, y, max(0, int(w * frac)), 12), border_radius=3)
        self.text(label, (x + w + 8, y - 2), TXT, self.f_small)

    def draw_status(self):
        y = GY + GH + 12
        m = self.m
        self.bar(GX, y, 160, m.health / C.MAX_HEALTH, GREEN if m.health > 40 else RED, f"Health {max(m.health, 0)}")
        self.bar(GX + 270, y, 160, m.battery / C.MAX_BATTERY, YEL, f"Battery {m.battery}")
        self.text(
            f"Rescued {len(m.rescued)}/{m.total_survivors}   Map known {self.bot.cmap.coverage():.0%}   "
            f"Step {m.steps}   Fire hits {m.fire_hits}",
            (GX, y + 22),
            TXT,
        )
        self.text(
            f"Mission #{self.episode}  map seed {self.seed}  |  speed {SPEEDS[self.speed_i]} steps/s  |  "
            f"learning {'ON' if self.learning else 'OFF'}  |  god-view {'ON' if self.god else 'OFF'}",
            (GX, y + 42),
            DIM,
            self.f_small,
        )
        self.text(
            "SPACE pause  N step  UP/DN speed  G god-view  H danger  R new map  L learn  S save  ESC quit",
            (GX, y + 60),
            DIM,
            self.f_small,
        )
        # legend
        lx, ly = GX, y + 82
        for col, name in [
            (ROBOT, "robot"),
            (SURV, "survivor"),
            (WALL, "debris"),
            (SMOKE, "smoke"),
            (FIRE, "fire"),
            (BASE_C, "base"),
            (PATH, "A* plan"),
            (FOG, "unexplored"),
        ]:
            pygame.draw.rect(self.screen, col, (lx, ly + 2, 12, 12), border_radius=2)
            self.text(name, (lx + 16, ly), DIM, self.f_small)
            lx += 78

    def draw_panel(self):
        pygame.draw.rect(self.screen, PANEL, (PX - 6, GY - 6, WIN_W - PX, WIN_H - GY), border_radius=8)
        t, x, y = self.trace, PX + 6, GY + 2
        self.text("COGNITIVE LOOP  (what the robot is thinking)", (x, y), TXT, self.f_head)
        y += 22
        if not t:
            self.text("Press SPACE / wait for the first step ...", (x, y), DIM)
            return

        def head(name, y):
            pygame.draw.rect(self.screen, STAGE_COL[name], (x, y + 3, 4, 13))
            self.text(name, (x + 10, y), STAGE_COL[name], self.f_head)
            return y + 18

        p = t["percept"]
        y = head("PERCEIVE", y)
        self.text(f"5x5 camera: {len(p.survivors)} survivor(s), {len(p.smoke)} smoky cell(s)", (x + 10, y))
        y += 16
        hot = [c for c, h in p.heat.items() if h]
        self.text(
            f"Heat sensor: {'HOT at ' + str(hot[0]) if hot else 'no heat nearby'}", (x + 10, y), RED if hot else TXT
        )
        y += 20

        y = head("REMEMBER", y)
        self.text("Working mem: " + " | ".join(self.bot.wm.slots[1:4]), (x + 10, y), TXT, self.f_small)
        y += 15
        self.text("Episodic (latest):", (x + 10, y), DIM, self.f_small)
        y += 15
        for e in self.bot.epi.recent(3)[::-1]:
            col = RED if e[2] == "burned" else GREEN if e[2] == "rescued" else TXT
            self.text("  " + e[3], (x + 10, y), col, self.f_small)
            y += 14
        self.text("Semantic (long-term):", (x + 10, y), DIM, self.f_small)
        y += 15
        for fct in self.sem.facts():
            self.text("  " + fct, (x + 10, y), (200, 180, 255), self.f_small)
            y += 14
        y += 6

        y = head("REASON", y)
        for r in t["fired"][:4]:
            self.text("  " + r, (x + 10, y), TXT, self.f_small)
            y += 14
        if not t["fired"]:
            self.text("  no rules fired", (x + 10, y), DIM, self.f_small)
            y += 14
        y += 6

        y = head("PLAN", y)
        tgt = t["target"]
        self.text(f"A* route to {tgt}: {max(len(t['path']) - 1, 0)} steps" if tgt else "no target", (x + 10, y))
        y += 20

        y = head("DECIDE", y)
        mx = max([v[0] + C.Q_SCALE * t["qvals"][k] for k, v in t["opts"].items()] + [1])
        for k, (u, tg, _) in t["opts"].items():
            score = u + C.Q_SCALE * t["qvals"][k]
            chosen = k == t["action"]
            col = STAGE_COL["DECIDE"] if chosen else (100, 106, 125)
            self.text(f"{k:8s}", (x + 10, y), TXT if tg else DIM, self.f_small)
            pygame.draw.rect(self.screen, (45, 50, 64), (x + 80, y + 2, 180, 11), border_radius=3)
            if tg:
                pygame.draw.rect(
                    self.screen, col, (x + 80, y + 2, max(2, int(180 * max(score, 0) / mx)), 11), border_radius=3
                )
            lab = f"U={u:4.2f} Q={t['qvals'][k]:+.1f}" if tg else "unavailable"
            self.text(lab + (" <" if chosen else ""), (x + 268, y), TXT if chosen else DIM, self.f_small)
            y += 17
        y += 4

        y = head("ACT", y)
        ev = ", ".join(e for e in t["events"] if e != "smoke_entered") or "moved"
        self.text(f"{t['action']} -> {self.m.pos}  ({ev})  reward {t['reward']:+.2f}", (x + 10, y))
        y += 22
        self.text(f"Q-learning state: {t['state']}   ({self.q.updates} updates so far)", (x + 6, y), DIM, self.f_small)
        y += 24
        self.text("KNOWLEDGE BASE - rules that fired this step", (x, y), DIM, self.f_head)
        y += 18
        for code in [r.split(":")[0] for r in t["fired"]][:5]:
            self.text(code + "  " + RULES[code].split("[")[0].strip(), (x + 6, y), (170, 176, 192), self.f_small)
            y += 14

    def draw(self):
        self.screen.fill(BG)
        self.text("CogniBot", (GX, 14), ROBOT, self.f_title)
        self.text(
            "A cognitive-architecture rescue robot  |  Perceive > Remember > Reason > Plan > Decide > Act > Learn",
            (GX + 110, 19),
            DIM,
            self.f_small,
        )
        self.draw_grid()
        self.draw_status()
        self.draw_panel()
        if self.m.done:
            s = pygame.Surface((GW, 70), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (GX, GY + GH // 2 - 35))
            col = GREEN if self.m.recovered else RED
            self.text(
                f"{self.m.outcome}: {len(self.m.rescued)}/{self.m.total_survivors} survivors rescued in "
                f"{self.m.steps} steps",
                (GX + 20, GY + GH // 2 - 22),
                col,
                self.f_head,
            )
            self.text("Next mission starts automatically (R = now)", (GX + 20, GY + GH // 2 + 2), TXT)
        if self.paused:
            self.text("PAUSED", (GX + GW - 70, 18), YEL, self.f_head)
        pygame.display.flip()

    # ------------------------------------------------------------------ main loop
    def run(self):
        acc = 0.0
        while True:
            dt = self.clock.tick(60) / 1000
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    if self.learning:
                        self.save()
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif e.key in (pygame.K_RIGHT, pygame.K_n) and self.paused:
                        self.tick()
                    elif e.key == pygame.K_UP:
                        self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
                    elif e.key == pygame.K_DOWN:
                        self.speed_i = max(self.speed_i - 1, 0)
                    elif e.key == pygame.K_g:
                        self.god = not self.god
                    elif e.key == pygame.K_h:
                        self.heat = not self.heat
                    elif e.key == pygame.K_r:
                        self.new_mission(random.randint(1, 99999))
                    elif e.key == pygame.K_s:
                        self.save()
                    elif e.key == pygame.K_l:
                        self.learning = not self.learning
                        self.bot.learn = self.sem.learn = self.learning
            if not self.paused:
                acc += dt
                while acc >= 1 / SPEEDS[self.speed_i] and not self.m.done:
                    self.tick()
                    acc -= 1 / SPEEDS[self.speed_i]
                if self.m.done:
                    acc = 0
                    self.end_timer += dt
                    if self.end_timer > 3:
                        self.new_mission(random.randint(1, 99999))
            self.draw()


def snapshot(args):
    """Headless screenshot for the report: python main.py --snapshot out.png --steps 40"""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    app = App(args)
    for _ in range(args.steps):
        if not app.m.done:
            app.tick()
    app.draw()
    pygame.image.save(app.screen, args.snapshot)
    pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CogniBot live demo")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="forget everything learned (empty brain)")
    ap.add_argument("--god", action="store_true", help="start with god-view on")
    ap.add_argument("--snapshot", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--steps", type=int, default=40, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.snapshot:
        snapshot(a)
    else:
        App(a).run()
