"""Train CogniBot, then compare it with baseline agents on unseen maps.

Usage:  python evaluate.py              (200 training missions, 100 test maps)
        python evaluate.py --quick      (fast check: 40 training, 20 test)
Outputs go to results/ and the trained "brain" goes to memory/.
"""

import argparse
import csv
import time
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cognibot.agent import CogniBot, RandomAgent, ReactiveAgent
from cognibot.learning import QLearner
from cognibot.memory import SemanticMemory
from cognibot.simulation import run_episode

ROOT = Path(__file__).resolve().parent
RES, MEM = ROOT / "results", ROOT / "memory"


def moving_avg(xs, k=20):
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - k + 1) : i + 1]
        out.append(sum(w) / len(w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.train, a.test = 40, 20
    RES.mkdir(exist_ok=True)
    MEM.mkdir(exist_ok=True)
    t_start = time.time()

    # ---------------- 1. TRAINING (learning from experience) ----------------
    sem, q = SemanticMemory(learn=True), QLearner(epsilon=0.3)
    bot = CogniBot(sem, q, learn=True, seed=42)
    curve = []
    print(f"Training CogniBot for {a.train} missions ...")
    for i in range(a.train):
        q.epsilon = max(0.02, 0.3 * (1 - i / (0.7 * a.train)))  # explore less over time
        r = run_episode(bot, 10_000 + i)  # training maps
        r.update(episode=i + 1, epsilon=round(q.epsilon, 3), p_fire=round(sem.p_fire_given_smoke(), 4))
        curve.append(r)
        if (i + 1) % 20 == 0:
            last = curve[-20:]
            print(
                f"  ep {i + 1:4d} | avg score {mean(x['score'] for x in last):6.2f} | "
                f"rescued {mean(x['rescued'] for x in last):.2f}/5 | "
                f"fire hits {mean(x['fire_hits'] for x in last):.2f} "
                f"| P(fire|smoke)={sem.p_fire_given_smoke():.2f}"
            )
    sem.save(MEM / "semantic_memory.json")
    q.save(MEM / "q_table.json")
    with open(RES / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)

    # ---------------- 2. TESTING on unseen maps ----------------
    trained_sem = SemanticMemory(MEM / "semantic_memory.json", learn=False)
    trained_q = QLearner(MEM / "q_table.json", epsilon=0.0)
    agents = {
        "Random walk": RandomAgent(seed=7),
        "Reactive (no memory)": ReactiveAgent(seed=7),
        "CogniBot (untrained)": CogniBot(SemanticMemory(learn=False), None, learn=False, seed=7),
        "CogniBot + memory learning": CogniBot(trained_sem, None, learn=False, seed=7),
        "CogniBot full (memory + Q)": CogniBot(trained_sem, trained_q, learn=False, seed=7),
    }
    rows, summary = [], []
    print(f"\nTesting all agents on {a.test} unseen maps ...")
    for label, ag in agents.items():
        res = [run_episode(ag, s) for s in range(1, a.test + 1)]
        for r in res:
            r["agent"] = label
        rows += res
        n_total = sum(r["total"] for r in res)
        s = {
            "Agent": label,
            "Rescue rate (%)": round(100 * sum(r["rescued"] for r in res) / n_total, 1),
            "All 5 rescued (%)": round(100 * mean(r["rescued"] == r["total"] for r in res), 1),
            "Fire hits / mission": round(mean(r["fire_hits"] for r in res), 2),
            "Robot recovered (%)": round(100 * mean(r["recovered"] for r in res), 1),
            "Robot destroyed (%)": round(100 * mean(r["outcome"] == "Robot destroyed" for r in res), 1),
            "Steps / mission": round(mean(r["steps"] for r in res), 1),
            "Mean score": round(mean(r["score"] for r in res), 2),
            "Score SD": round(pstdev(r["score"] for r in res), 2),
            "Decision time (ms)": round(mean(r["decision_ms"] for r in res), 3),
        }
        summary.append(s)
        print("  " + " | ".join(f"{k}: {v}" for k, v in s.items()))

    with open(RES / "test_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(RES / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    md = ["| " + " | ".join(summary[0].keys()) + " |", "|" + "---|" * len(summary[0])]
    md += ["| " + " | ".join(str(v) for v in s.values()) + " |" for s in summary]
    (RES / "summary.md").write_text(
        f"# CogniBot evaluation\n\nTraining missions: {a.train}  |  "
        f"Test maps: {a.test} (seeds 1-{a.test}, never seen in training)\n\n" + "\n".join(md) + "\n"
    )

    # ---------------- 3. PLOTS ----------------
    plt.rcParams.update({"font.size": 10})
    ep = [c["episode"] for c in curve]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(ep, [c["score"] for c in curve], color="#bbb", lw=0.8, label="per mission")
    ax[0].plot(ep, moving_avg([c["score"] for c in curve]), color="#1f6feb", lw=2, label="20-mission average")
    ax[0].set_title("Mission score while learning")
    ax[0].set_xlabel("Training mission")
    ax[0].legend()
    ax[1].plot(ep, moving_avg([c["fire_hits"] for c in curve]), color="#d9480f", lw=2)
    ax[1].set_title("Fire hits per mission (20-avg)")
    ax[1].set_xlabel("Training mission")
    ax[2].plot(ep, [c["p_fire"] for c in curve], color="#2b8a3e", lw=2)
    ax[2].set_title("Learned belief P(fire | smoke)")
    ax[2].set_xlabel("Training mission")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "learning_curve.png", dpi=150)
    plt.close(fig)

    short = {
        "Random walk": "Random\nwalk",
        "Reactive (no memory)": "Reactive\n(no memory)",
        "CogniBot (untrained)": "CogniBot\nuntrained",
        "CogniBot + memory learning": "CogniBot\n+ memory",
        "CogniBot full (memory + Q)": "CogniBot\nfull (+Q)",
    }
    names = [short.get(s["Agent"], s["Agent"]) for s in summary]
    colors = ["#adb5bd", "#f08c00", "#a5d8ff", "#4dabf7", "#1c4fa3"]
    metrics = [
        ("Rescue rate (%)", "Survivors rescued (%)"),
        ("Fire hits / mission", "Fire hits per mission"),
        ("Robot recovered (%)", "Robot returned safely (%)"),
        ("Mean score", "Mean mission score"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for x, (k, title) in zip(ax, metrics):
        vals = [s[k] for s in summary]
        bars = x.bar(names, vals, color=colors)
        x.bar_label(bars, fmt="%.2f" if "Fire" in k else "%.1f", fontsize=9)
        x.set_title(title)
        x.tick_params(axis="x", labelsize=9)
        x.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Agent comparison on {a.test} unseen maps", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RES / "agent_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nDone in {time.time() - t_start:.1f}s. Results saved in {RES}")


if __name__ == "__main__":
    main()
