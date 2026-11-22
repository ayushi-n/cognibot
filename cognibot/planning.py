"""PLAN: Dijkstra cost-map (to evaluate goals) and A* search (to plan the route)."""

import heapq

from . import config as C
from .world import manhattan, nbrs4


def cell_cost(c, cmap, danger):
    if cmap.is_wall(c):
        return None
    base = 1.0 if c in cmap.known else C.UNKNOWN_COST
    return base + C.DANGER_WEIGHT * danger.get(c, 0.0)


def dijkstra(start, cmap, danger):
    """Cost of the cheapest route from the robot to every reachable cell."""
    dist, pq = {start: 0.0}, [(0.0, start)]
    while pq:
        d, c = heapq.heappop(pq)
        if d > dist[c]:
            continue
        for n in nbrs4(c):
            k = cell_cost(n, cmap, danger)
            if k is not None and d + k < dist.get(n, 1e18):
                dist[n] = d + k
                heapq.heappush(pq, (d + k, n))
    return dist


def astar(start, goal, cmap, danger):
    """A* with Manhattan heuristic (admissible because every step costs >= 1)."""
    if goal is None:
        return None
    g, parent = {start: 0.0}, {start: None}
    pq = [(manhattan(start, goal), 0.0, start)]
    while pq:
        _, d, c = heapq.heappop(pq)
        if c == goal:
            path = []
            while c is not None:
                path.append(c)
                c = parent[c]
            return path[::-1]
        if d > g[c]:
            continue
        for n in nbrs4(c):
            k = cell_cost(n, cmap, danger)
            if k is not None and d + k < g.get(n, 1e18):
                g[n], parent[n] = d + k, c
                heapq.heappush(pq, (d + k + manhattan(n, goal), d + k, n))
    return None
