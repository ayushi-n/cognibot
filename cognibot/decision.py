"""DECIDE: utility-based scoring of the three high-level options."""

from . import config as C

OPTIONS = ("RESCUE", "EXPLORE", "RETREAT")


def u_rescue(cost):
    return 60.0 / (cost + 5)  # value of a life / effort


def u_explore(gain, cost):
    return 0.8 * gain / (cost + 3)  # information gain / effort


def evaluate_options(pos, cmap, dist, flags, health, battery, steps):
    """Returns {option: (utility, target_cell, path_cost)}. target None = unavailable."""
    opts = {}
    best = (0.0, None, None)
    for s in cmap.survivors:
        if s in dist:
            u = u_rescue(dist[s])
            if u > best[0]:
                best = (u, s, dist[s])
    opts["RESCUE"] = best

    best = (0.0, None, None)
    for f in cmap.frontier():
        if f == pos or f not in dist:
            continue
        gain = cmap.unknown_around(f)
        if gain == 0:
            continue
        u = u_explore(gain, dist[f])
        if u > best[0]:
            best = (u, f, dist[f])
    opts["EXPLORE"] = best

    cb = dist.get(C.BASE)
    u = 6.0 * (1 - health / C.MAX_HEALTH) ** 2  # grows as the robot gets hurt
    if flags["low_health"]:
        u += 3.0
    if flags["low_battery"]:
        u += 10.0
    if flags.get("all_rescued"):
        opts["RESCUE"], opts["EXPLORE"] = (0.0, None, None), (0.0, None, None)
    nothing_left = opts["RESCUE"][1] is None and opts["EXPLORE"][1] is None
    if nothing_left:
        u += 5.0
    available = cb is not None and (pos != C.BASE or nothing_left or flags["low_battery"] or flags["low_health"])
    opts["RETREAT"] = (u, C.BASE if available else None, cb)
    return opts
