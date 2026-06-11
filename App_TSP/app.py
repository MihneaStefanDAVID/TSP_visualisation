import math
import json
import random
import urllib.request
import urllib.parse
import urllib.error
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Color palette (kept in sync with the frontend)
# ---------------------------------------------------------------------------
COLOR_NODE_DEFAULT = "#2b2a26"
COLOR_NODE_VISITED = "#8a8578"
COLOR_NODE_ACTIVE = "#a07b1a"
COLOR_NODE_ODD = "#9c4a1b"
COLOR_BLUE = "#1e4d8c"
COLOR_GREEN = "#2f5d3e"
COLOR_ORANGE = "#9c4a1b"
COLOR_PURPLE = "#5d3a8c"
COLOR_YELLOW = "#a07b1a"
COLOR_RED = "#8b1a1a"
COLOR_WHITE = "#faf7ee"

HELD_KARP_MAX_N = 20  # cap for exact optimum

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
state = {
    "nodes": [],
    "dist": [],
    "optimal_cost": None,
    "optimal_exact": False,
    "last_steps": None,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def euclidean(p1, p2):
    return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])


def build_distance_matrix(nodes):
    n = len(nodes)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = euclidean(nodes[i], nodes[j])
            d[i][j] = v
            d[j][i] = v
    return d


def tour_cost(cycle, dist):
    if len(cycle) < 2:
        return 0.0
    total = 0.0
    for i in range(len(cycle) - 1):
        total += dist[cycle[i]][cycle[i + 1]]
    return total


def prims_mst(nodes, dist):
    n = len(nodes)
    if n == 0:
        return []
    in_tree = [False] * n
    min_edge = [math.inf] * n
    parent = [-1] * n
    min_edge[0] = 0.0
    edges = []
    for _ in range(n):
        u = -1
        best = math.inf
        for v in range(n):
            if not in_tree[v] and min_edge[v] < best:
                best = min_edge[v]
                u = v
        if u == -1:
            break
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u))
        for v in range(n):
            if not in_tree[v] and dist[u][v] < min_edge[v]:
                min_edge[v] = dist[u][v]
                parent[v] = u
    return edges


def hierholzers_eulerian(n, multi_edges, start=0):
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(multi_edges):
        adj[u].append([v, idx])
        adj[v].append([u, idx])
    used = [False] * len(multi_edges)

    stack = [start]
    path = []
    ptr = [0] * n
    while stack:
        u = stack[-1]
        while ptr[u] < len(adj[u]) and used[adj[u][ptr[u]][1]]:
            ptr[u] += 1
        if ptr[u] == len(adj[u]):
            path.append(stack.pop())
        else:
            v, eid = adj[u][ptr[u]]
            used[eid] = True
            ptr[u] += 1
            stack.append(v)
    path.reverse()
    return path


def shortcut_events(euler):
    """Walk the Eulerian circuit and emit one event per newly-added node in the
    final Hamiltonian cycle. Each event records which subpath of the Eulerian
    circuit was replaced by a single direct edge.

    Returns (hamiltonian_cycle, events). events is a list 
    of dicts with:
        from        : node u (last committed vertex)
        to          : node v (next unvisited vertex)
        subpath     : list [u, ..., v] of the original Eulerian subpath
        skipped     : list of intermediate vertices that were revisited
        closing     : True if this event closes the cycle back to start
    """
    if not euler:
        return [], []

    events = []
    seen = set()
    ham = [euler[0]]
    seen.add(euler[0])
    last_added = euler[0]
    subpath = [euler[0]]

    for i in range(1, len(euler)):
        v = euler[i]
        subpath.append(v)
        is_last = (i == len(euler) - 1)
        # new unvisited node -> commit it
        if v not in seen:
            skipped = subpath[1:-1]
            events.append({
                "from": last_added,
                "to": v,
                "subpath": list(subpath),
                "skipped": list(skipped),
                "closing": False,
            })
            ham.append(v)
            seen.add(v)
            last_added = v
            subpath = [v]
        elif is_last:
            # closing the cycle
            start = ham[0]
            skipped = subpath[1:-1]
            events.append({
                "from": last_added,
                "to": start,
                "subpath": list(subpath),
                "skipped": list(skipped),
                "closing": True,
            })
            ham.append(start)

    return ham, events


def compute_euler_offsets(euler_edges, spacing=8):
    """For each Eulerian edge position, compute a perpendicular offset so that
    multiple edges between the same pair of nodes are drawn as parallel lines.
    The offset parameter must respect the drawn direction so that opposite-
    direction copies end up on different visual sides rather than overlapping.
    """
    from collections import defaultdict
    pair_positions = defaultdict(list)
    for idx, (a, b) in enumerate(euler_edges):
        key = (min(a, b), max(a, b))
        pair_positions[key].append(idx)
    offs = [0.0] * len(euler_edges)
    for key, positions in pair_positions.items():
        k = len(positions)
        if k <= 1:
            continue
        for slot, idx in enumerate(positions):
            world_offset = (slot - (k - 1) / 2) * spacing
            u, v = euler_edges[idx]
            sign = 1 if u < v else -1
            offs[idx] = world_offset * sign
    return offs


def greedy_matching(odd_nodes, dist):
    pairs_by_d = []
    for i in range(len(odd_nodes)):
        for j in range(i + 1, len(odd_nodes)):
            u = odd_nodes[i]
            v = odd_nodes[j]
            pairs_by_d.append((dist[u][v], u, v))
    pairs_by_d.sort()
    used = set()
    matching = []
    for d, u, v in pairs_by_d:
        if u in used or v in used:
            continue
        used.add(u)
        used.add(v)
        matching.append((u, v))
    return matching


def held_karp(dist):
    """Exact TSP via Held-Karp DP, vectorized with numpy.

    dp[mask, j] = shortest path starting at 0, visiting exactly vertices in
    mask, ending at j. Transition per mask:
        cand[k] = min_{j in mask} dp[mask, j] + D[j, k]      for k not in mask
    This inner reduction is a vectorized (n×n) broadcast → min, so the
    per-mask work is O(n²) in tight numpy kernels rather than interpreted
    bit loops. That makes n=20 feasible (≈4 s/instance).
    """
    n = len(dist)
    if n <= 1:
        return 0.0
    if n > HELD_KARP_MAX_N:
        return None
    D = np.asarray(dist, dtype=np.float64)
    size = 1 << n
    dp = np.full((size, n), np.inf, dtype=np.float64)
    dp[1, 0] = 0.0
    all_bits = size - 1

    for mask in range(1, size):
        if not (mask & 1):
            continue
        row = dp[mask]
        # cand[k] = min_j (row[j] + D[j,k])
        cand = (row[:, None] + D).min(axis=0)
        rem = all_bits & ~mask
        m = rem
        while m:
            lsb = m & -m
            k = lsb.bit_length() - 1
            m ^= lsb
            nm = mask | lsb
            if cand[k] < dp[nm, k]:
                dp[nm, k] = cand[k]

    best = float(np.min(dp[all_bits, 1:] + D[1:, 0]))
    return best


# ---------------------------------------------------------------------------
# Step rendering helpers
# ---------------------------------------------------------------------------
def base_nodes(color_map=None, highlight_set=None, border_map=None):
    color_map = color_map or {}
    highlight_set = highlight_set or set()
    border_map = border_map or {}
    out = []
    for node in state["nodes"]:
        nid = node["id"]
        out.append({
            "id": nid,
            "x": node["x"],
            "y": node["y"],
            "color": color_map.get(nid, COLOR_NODE_DEFAULT),
            "highlight": nid in highlight_set,
            "border": border_map.get(nid),
        })
    return out


def edge_dict(u, v, color, width=2, dashed=False, label=None, offset=0, arrow=False, order=None):
    return {
        "from": u,
        "to": v,
        "color": color,
        "width": width,
        "dashed": dashed,
        "label": label,
        "offset": offset,
        "arrow": arrow,
        "order": order,
    }


def fmt(x):
    if x is None:
        return None
    return round(float(x), 2)


def build_stats(algo, tour_cost_val=None, mst_cost=None, matching_cost=None):
    """Return an ordered list of {label, value} pairs for the side panel."""
    opt = state["optimal_cost"]
    opt_exact = state["optimal_exact"]
    rows = []

    # Shared header: tour cost
    rows.append({"label": "Tour cost",
                 "value": None if tour_cost_val is None else f"{tour_cost_val:.2f}"})

    if algo in ("mst", "christofides"):
        rows.append({"label": "MST lower bound",
                     "value": None if mst_cost is None else f"{mst_cost:.2f}"})
    if algo == "christofides":
        rows.append({"label": "Matching cost",
                     "value": None if matching_cost is None else f"{matching_cost:.2f}"})

    # Optimal cost
    if opt is None:
        rows.append({"label": "Optimal cost",
                     "value": f"— (n > {HELD_KARP_MAX_N}, exact solver skipped)"})
    else:
        label = "Optimal cost" if opt_exact else "Optimal cost (approx.)"
        rows.append({"label": label, "value": f"{opt:.2f}"})

    # Ratio vs optimal
    if tour_cost_val is not None and opt is not None and opt > 0:
        rows.append({"label": "Ratio (tour / OPT)",
                     "value": f"{tour_cost_val / opt:.3f}"})
    else:
        rows.append({"label": "Ratio (tour / OPT)", "value": None})

    # Ratio vs MST (for MST-based algos only)
    if algo in ("mst", "christofides") and tour_cost_val is not None \
            and mst_cost is not None and mst_cost > 0:
        rows.append({"label": "Ratio (tour / MST)",
                     "value": f"{tour_cost_val / mst_cost:.3f}"})

    # Theoretical guarantee
    guarantees = {
        "nn": "No constant guarantee (Θ(log n))",
        "mst": "≤ 2 · OPT",
        "christofides": "≤ 3/2 · OPT",
        "ni": "≤ 2 · OPT",
    }
    rows.append({"label": "Guarantee", "value": guarantees.get(algo, "—")})

    return rows


# ---------------------------------------------------------------------------
# Shortcut animation helper (shared by MST 2-approx and Christofides)
# ---------------------------------------------------------------------------
def build_shortcut_steps(euler_path, dist, mst_edges_list, matching_edges_list,
                         algo, mst_cost_val, matching_cost_val,
                         step_start_idx):
    """Return a list of step dicts animating the shortcut phase.

    Visualization per step:
      - previously-committed Hamiltonian edges  -> purple (thick)
      - edges of the Eulerian subpath being replaced -> red dashed
      - the newly-inserted shortcut edge        -> yellow (thick)
      - remaining not-yet-traversed Eulerian edges -> faint orange
    """
    steps = []
    ham, events = shortcut_events(euler_path)
    n = len(state["nodes"])

    # Precompute Euler edge index map (position -> (a,b))
    euler_edges = [(euler_path[i], euler_path[i + 1])
                   for i in range(len(euler_path) - 1)]
    euler_offsets = compute_euler_offsets(euler_edges)

    # Track which euler edge positions have already been "consumed" by events
    consumed_upto = 0
    committed_ham_edges = []  # list of (u,v) forming final ham so far
    visited_so_far = {euler_path[0]}
    step_idx = step_start_idx

    # Intro step for the shortcut phase
    # Show euler in light orange + start node highlighted
    intro_edges = []
    for i, (a, b) in enumerate(euler_edges):
        intro_edges.append(edge_dict(a, b, "#9c4a1b70", width=2, arrow=True,
                                     offset=euler_offsets[i]))
    color_map = {euler_path[0]: COLOR_YELLOW}
    intro_stats = build_stats(algo, tour_cost_val=0.0, mst_cost=mst_cost_val,
                              matching_cost=matching_cost_val)
    steps.append({
        "title": f"Step {step_idx}: Shortcutting phase — begin",
        "description": (
            "We now walk the Eulerian circuit and build a Hamiltonian cycle by "
            "skipping every vertex we have already visited. At each step the "
            "edges shown in red (dashed) are the Eulerian subpath we are about "
            "to bypass; the yellow edge is the direct shortcut that replaces "
            "them. By the triangle inequality, each such shortcut can only "
            "decrease the total length. The running Hamiltonian cycle is drawn "
            "in purple."
        ),
        "nodes": base_nodes(color_map=color_map, highlight_set={euler_path[0]}),
        "edges": intro_edges,
        "stats": intro_stats,
    })
    step_idx += 1

    # Process each event
    for ev_idx, ev in enumerate(events):
        subpath = ev["subpath"]
        skipped = ev["skipped"]
        u = ev["from"]
        v = ev["to"]

        # Count subpath edges
        sub_len = len(subpath) - 1
        # indices of the euler edges being consumed by this event
        replaced_indices = list(range(consumed_upto, consumed_upto + sub_len))
        replaced = [euler_edges[i] for i in replaced_indices]
        consumed_upto += sub_len

        # Cost of replaced euler subpath
        replaced_cost = sum(dist[a][b] for (a, b) in replaced)
        direct_cost = dist[u][v]

        # Build visualization
        edges_vis = []

        # Remaining euler (faint orange) - edges not yet consumed
        for i in range(consumed_upto, len(euler_edges)):
            a, b = euler_edges[i]
            edges_vis.append(edge_dict(a, b, "#9c4a1b50", width=2, arrow=True,
                                       offset=euler_offsets[i]))

        # Replaced subpath edges in red dashed (keep their offset so copies stay parallel)
        for i in replaced_indices:
            a, b = euler_edges[i]
            edges_vis.append(edge_dict(a, b, COLOR_RED, width=2, dashed=True,
                                       label=f"{dist[a][b]:.1f}",
                                       offset=euler_offsets[i]))

        # Previously committed ham edges in purple
        for (a, b) in committed_ham_edges:
            edges_vis.append(edge_dict(a, b, COLOR_PURPLE, width=3))

        # New shortcut edge in yellow (this step's result)
        edges_vis.append(edge_dict(u, v, COLOR_YELLOW, width=4,
                                   label=f"{direct_cost:.1f}"))

        # Node coloring
        color_map = {}
        # visited so far -> dim
        for nid in visited_so_far:
            color_map[nid] = COLOR_NODE_VISITED
        # skipped intermediates in this event -> red-ish
        for nid in skipped:
            color_map[nid] = COLOR_RED
        # start/prev node -> neutral-highlighted
        color_map[u] = COLOR_NODE_VISITED
        # new node -> yellow
        color_map[v] = COLOR_YELLOW
        # but if closing (returning to start), highlight start in yellow too
        if ev["closing"]:
            color_map[v] = COLOR_YELLOW

        # Description
        if len(skipped) == 0:
            desc = (
                f"Moving from node {u} to the next unvisited node {v}. "
                f"No revisited vertices to skip — the direct edge (u, v) is "
                f"identical to the Eulerian edge. Added to the Hamiltonian "
                f"cycle (purple) with length {direct_cost:.2f}."
            )
        else:
            skipped_str = ", ".join(str(s) for s in skipped)
            desc = (
                f"From node {u}, the Eulerian circuit passes through already-visited "
                f"nodes [{skipped_str}] before reaching the next unvisited node {v}. "
                f"We bypass that subpath (red dashed, total length {replaced_cost:.2f}) "
                f"and insert the direct edge ({u}, {v}) of length {direct_cost:.2f} "
                f"(yellow). By the triangle inequality d({u},{v}) ≤ "
                f"d({u},·) + … + d(·,{v}), so this shortcut does not increase cost."
            )
        if ev["closing"]:
            desc += " This shortcut closes the Hamiltonian cycle back to the starting vertex."

        # Running tour cost so far (committed + this new edge)
        running_cost = (sum(dist[a][b] for (a, b) in committed_ham_edges)
                        + direct_cost)

        # commit this event's edge
        committed_ham_edges.append((u, v))
        visited_so_far.add(v)

        stats = build_stats(algo, tour_cost_val=running_cost,
                            mst_cost=mst_cost_val,
                            matching_cost=matching_cost_val)
        steps.append({
            "title": f"Step {step_idx}: Shortcut → node {v}"
                     + (" (close cycle)" if ev["closing"] else ""),
            "description": desc,
            "nodes": base_nodes(color_map=color_map, highlight_set={v}),
            "edges": edges_vis,
            "stats": stats,
        })
        step_idx += 1

    # Final clean step: full Hamiltonian in purple
    ham_cost = tour_cost(ham, dist)
    final_edges = [edge_dict(ham[i], ham[i + 1], COLOR_PURPLE, width=3,
                             label=f"{dist[ham[i]][ham[i + 1]]:.1f}")
                   for i in range(len(ham) - 1)]
    final_stats = build_stats(algo, tour_cost_val=ham_cost,
                              mst_cost=mst_cost_val,
                              matching_cost=matching_cost_val)
    if algo == "mst":
        final_desc = (
            f"All shortcuts applied. The final Hamiltonian cycle has length "
            f"ℓ(C) = {ham_cost:.2f}. Since ℓ(C) ≤ 2 · ℓ(T) ≤ 2 · OPT, this is "
            f"the MST-doubling 2-approximation."
        )
    else:
        final_desc = (
            f"All shortcuts applied. The final Hamiltonian cycle has length "
            f"ℓ(C) = {ham_cost:.2f}. Since ℓ(C) ≤ ℓ(T) + ℓ(M) ≤ OPT + ½·OPT, "
            f"this is the Christofides 3/2-approximation."
        )
    steps.append({
        "title": f"Step {step_idx}: Final Hamiltonian cycle",
        "description": final_desc,
        "nodes": base_nodes(),
        "edges": final_edges,
        "stats": final_stats,
    })
    return steps, ham, ham_cost


# ---------------------------------------------------------------------------
# Algorithm: Nearest Neighbor
# ---------------------------------------------------------------------------
def algo_nearest_neighbor():
    nodes = state["nodes"]
    dist = state["dist"]
    n = len(nodes)
    steps = []

    steps.append({
        "title": "Step 1: Nearest Neighbor — Setup",
        "description": (
            "The Nearest Neighbor heuristic is the most intuitive TSP approximation: "
            "starting from an arbitrary vertex, repeatedly walk to the closest unvisited "
            "neighbor until every vertex has been visited, then return to the start. "
            "Although conceptually simple, this algorithm has NO constant-factor approximation "
            "guarantee — its worst-case ratio grows as Θ(log n). We begin at node 0 "
            "(highlighted in yellow)."
        ),
        "nodes": base_nodes(highlight_set={0}, color_map={0: COLOR_YELLOW}),
        "edges": [],
        "stats": build_stats("nn"),
    })

    visited = [0]
    visited_set = {0}
    tour_edges = []
    current = 0
    step_idx = 2
    while len(visited) < n:
        best = -1
        best_d = math.inf
        for v in range(n):
            if v not in visited_set and dist[current][v] < best_d:
                best_d = dist[current][v]
                best = v
        tour_edges.append((current, best))
        visited_set.add(best)
        visited.append(best)

        color_map = {nid: COLOR_NODE_VISITED for nid in visited_set}
        color_map[current] = COLOR_YELLOW
        color_map[best] = COLOR_YELLOW

        edges_out = []
        for i, (u, v) in enumerate(tour_edges):
            if i == len(tour_edges) - 1:
                edges_out.append(edge_dict(u, v, COLOR_ORANGE, width=3,
                                           label=f"{dist[u][v]:.1f}"))
            else:
                edges_out.append(edge_dict(u, v, COLOR_PURPLE, width=2))

        partial = tour_cost(visited, dist)
        steps.append({
            "title": f"Step {step_idx}: Visit node {best}",
            "description": (
                f"From node {current}, the nearest unvisited neighbor is node {best} "
                f"at Euclidean distance {best_d:.2f}. We traverse that edge (orange). "
                f"Partial tour length: {partial:.2f}."
            ),
            "nodes": base_nodes(color_map=color_map, highlight_set={best, current}),
            "edges": edges_out,
            "stats": build_stats("nn", tour_cost_val=partial),
        })
        current = best
        step_idx += 1

    tour_edges.append((current, 0))
    cycle = visited + [0]
    total = tour_cost(cycle, dist)
    color_map = {nid: COLOR_NODE_VISITED for nid in range(n)}
    color_map[0] = COLOR_YELLOW
    edges_out = [edge_dict(u, v, COLOR_PURPLE, width=3) for (u, v) in tour_edges]
    steps.append({
        "title": f"Step {step_idx}: Final tour",
        "description": (
            f"All {n} vertices visited; we close the tour by returning to node 0. "
            f"Total length: {total:.2f}. Nearest Neighbor has no constant "
            f"approximation ratio — worst-case Θ(log n)."
        ),
        "nodes": base_nodes(color_map=color_map, highlight_set={0}),
        "edges": edges_out,
        "stats": build_stats("nn", tour_cost_val=total),
    })
    return steps


# ---------------------------------------------------------------------------
# Algorithm: MST 2-approximation
# ---------------------------------------------------------------------------
def algo_mst_2approx():
    nodes = state["nodes"]
    dist = state["dist"]
    n = len(nodes)
    steps = []

    steps.append({
        "title": "Step 1: Setup — Metric TSP",
        "description": (
            "The MST-based 2-approximation exploits a structural fact about metric TSP: "
            "a minimum spanning tree T has weight ℓ(T) ≤ OPT, because removing any edge "
            "from an optimal Hamiltonian cycle yields a spanning tree. We'll use T as a "
            "lower bound. This algorithm requires the triangle inequality."
        ),
        "nodes": base_nodes(),
        "edges": [],
        "stats": build_stats("mst"),
    })

    mst_edges = prims_mst(nodes, dist)
    mst_cost = sum(dist[u][v] for (u, v) in mst_edges)
    edges_mst = [edge_dict(u, v, COLOR_BLUE, width=3, label=f"{dist[u][v]:.1f}")
                 for (u, v) in mst_edges]
    steps.append({
        "title": "Step 2: Build the Minimum Spanning Tree",
        "description": (
            f"Using Prim's algorithm we compute an MST with n − 1 = {n - 1} edges and "
            f"total weight ℓ(T) = {mst_cost:.2f}. This is a provable lower bound on OPT."
        ),
        "nodes": base_nodes(),
        "edges": edges_mst,
        "stats": build_stats("mst", mst_cost=mst_cost),
    })

    # Doubled MST
    doubled_vis = []
    doubled_multi = []
    for (u, v) in mst_edges:
        doubled_vis.append(edge_dict(u, v, COLOR_GREEN, width=2, offset=-4))
        doubled_vis.append(edge_dict(u, v, COLOR_GREEN, width=2, offset=4))
        doubled_multi.append((u, v))
        doubled_multi.append((u, v))
    steps.append({
        "title": "Step 3: Double every edge",
        "description": (
            "Create a multigraph T' by doubling every edge of T. Every vertex in T' now "
            "has even degree, so an Eulerian circuit exists. Total weight of T' = "
            f"2 · ℓ(T) = {2 * mst_cost:.2f} ≤ 2 · OPT."
        ),
        "nodes": base_nodes(),
        "edges": doubled_vis,
        "stats": build_stats("mst", mst_cost=mst_cost),
    })

    euler = hierholzers_eulerian(n, doubled_multi, start=0)
    euler_edges_full = [(euler[i], euler[i + 1]) for i in range(len(euler) - 1)]
    euler_offsets_full = compute_euler_offsets(euler_edges_full)
    euler_vis = []
    for i, (a, b) in enumerate(euler_edges_full):
        euler_vis.append(edge_dict(
            a, b, COLOR_ORANGE, width=2, arrow=True,
            label=str(i + 1), order=i + 1,
            offset=euler_offsets_full[i],
        ))
    steps.append({
        "title": "Step 4: Eulerian circuit via Hierholzer's algorithm",
        "description": (
            f"Hierholzer's algorithm produces an Eulerian circuit in O(|E|) time. "
            f"It visits {len(euler) - 1} edges (each MST edge twice). Numbers indicate "
            f"traversal order."
        ),
        "nodes": base_nodes(),
        "edges": euler_vis,
        "stats": build_stats("mst", mst_cost=mst_cost),
    })

    # Step-by-step shortcut animation
    shortcut_steps, ham, ham_cost = build_shortcut_steps(
        euler, dist, mst_edges, [], "mst", mst_cost, None, step_start_idx=5
    )
    steps.extend(shortcut_steps)
    return steps


# ---------------------------------------------------------------------------
# Algorithm: Christofides
# ---------------------------------------------------------------------------
def algo_christofides():
    nodes = state["nodes"]
    dist = state["dist"]
    n = len(nodes)
    steps = []

    steps.append({
        "title": "Step 1: Christofides–Serdyukov — Overview",
        "description": (
            "Christofides (1976) improves the MST 2-approximation to a 3/2-approximation "
            "for metric TSP. Instead of doubling ALL MST edges (cost 2·ℓ(T)), we only add "
            "edges to fix the parity of odd-degree vertices, using a minimum-weight "
            "perfect matching M. Total cost: ℓ(T) + ℓ(M) ≤ OPT + ½·OPT = 3/2·OPT."
        ),
        "nodes": base_nodes(),
        "edges": [],
        "stats": build_stats("christofides"),
    })

    mst_edges = prims_mst(nodes, dist)
    mst_cost = sum(dist[u][v] for (u, v) in mst_edges)
    mst_vis = [edge_dict(u, v, COLOR_BLUE, width=3, label=f"{dist[u][v]:.1f}")
               for (u, v) in mst_edges]
    steps.append({
        "title": "Step 2: Compute the MST",
        "description": f"MST T has weight ℓ(T) = {mst_cost:.2f}. ℓ(T) ≤ OPT.",
        "nodes": base_nodes(),
        "edges": mst_vis,
        "stats": build_stats("christofides", mst_cost=mst_cost),
    })

    deg = [0] * n
    for (u, v) in mst_edges:
        deg[u] += 1
        deg[v] += 1
    odd = [i for i in range(n) if deg[i] % 2 == 1]
    odd_set = set(odd)
    color_map = {i: COLOR_ORANGE for i in odd}
    steps.append({
        "title": "Step 3: Identify odd-degree vertices",
        "description": (
            f"In T, {len(odd)} vertices have odd degree (orange). By the Handshaking "
            f"Lemma the count is always even, so a perfect matching on them always exists."
        ),
        "nodes": base_nodes(color_map=color_map, highlight_set=odd_set),
        "edges": mst_vis,
        "stats": build_stats("christofides", mst_cost=mst_cost),
    })

    matching = greedy_matching(odd, dist)
    match_cost = sum(dist[u][v] for (u, v) in matching)
    match_vis = [edge_dict(u, v, COLOR_GREEN, width=3, dashed=True,
                           label=f"{dist[u][v]:.1f}") for (u, v) in matching]
    steps.append({
        "title": "Step 4: Perfect matching on odd vertices",
        "description": (
            f"We compute a perfect matching M on the odd-degree vertices (dashed green). "
            f"Edmonds' blossom algorithm finds a true minimum; here we use a greedy "
            f"approximation for clarity. Weight: ℓ(M) = {match_cost:.2f}. "
            f"Theoretical bound: ℓ(M*) ≤ ½ · OPT."
        ),
        "nodes": base_nodes(color_map=color_map, highlight_set=odd_set),
        "edges": mst_vis + match_vis,
        "stats": build_stats("christofides", mst_cost=mst_cost,
                             matching_cost=match_cost),
    })

    steps.append({
        "title": "Step 5: Multigraph H = T ∪ M",
        "description": (
            "Superimposing T (blue) and M (green) gives a multigraph H. Every vertex in H "
            "has even degree, so H admits an Eulerian circuit. "
            f"Total weight = ℓ(T) + ℓ(M) = {mst_cost + match_cost:.2f}."
        ),
        "nodes": base_nodes(color_map=color_map, highlight_set=odd_set),
        "edges": mst_vis + match_vis,
        "stats": build_stats("christofides", mst_cost=mst_cost,
                             matching_cost=match_cost),
    })

    multi_edges = list(mst_edges) + list(matching)
    euler = hierholzers_eulerian(n, multi_edges, start=0)
    euler_edges_full = [(euler[i], euler[i + 1]) for i in range(len(euler) - 1)]
    euler_offsets_full = compute_euler_offsets(euler_edges_full)
    euler_vis = []
    for i, (a, b) in enumerate(euler_edges_full):
        euler_vis.append(edge_dict(
            a, b, COLOR_ORANGE, width=2, arrow=True,
            label=str(i + 1), order=i + 1,
            offset=euler_offsets_full[i],
        ))
    steps.append({
        "title": "Step 6: Eulerian circuit of H",
        "description": (
            f"Hierholzer's algorithm produces an Eulerian circuit of H visiting "
            f"{len(euler) - 1} edges. Its weight equals ℓ(T) + ℓ(M) ≤ 3/2 · OPT."
        ),
        "nodes": base_nodes(),
        "edges": euler_vis,
        "stats": build_stats("christofides", mst_cost=mst_cost,
                             matching_cost=match_cost),
    })

    shortcut_steps, ham, ham_cost = build_shortcut_steps(
        euler, dist, mst_edges, matching, "christofides",
        mst_cost, match_cost, step_start_idx=7
    )
    steps.extend(shortcut_steps)
    return steps


# ---------------------------------------------------------------------------
# Algorithm: Nearest Insertion
# ---------------------------------------------------------------------------
def algo_nearest_insertion():
    nodes = state["nodes"]
    dist = state["dist"]
    n = len(nodes)
    steps = []

    best_pair = (0, 1)
    best_d = math.inf
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] < best_d:
                best_d = dist[i][j]
                best_pair = (i, j)
    a, b = best_pair
    tour = [a, b, a]

    color_map = {a: COLOR_YELLOW, b: COLOR_YELLOW}
    init_edges = [
        edge_dict(a, b, COLOR_PURPLE, width=3, label=f"{dist[a][b]:.1f}"),
        edge_dict(b, a, COLOR_PURPLE, width=3),
    ]
    steps.append({
        "title": "Step 1: Nearest Insertion — Initialize",
        "description": (
            f"Start with the two closest vertices {a} and {b} (distance {dist[a][b]:.2f}), "
            f"forming a trivial 2-cycle. At each subsequent step: (1) pick the vertex "
            f"outside the tour that is closest to any vertex in the tour; (2) insert it "
            f"where it causes the least increase in tour length. Nearest Insertion is a "
            f"2-approximation for metric TSP."
        ),
        "nodes": base_nodes(color_map=color_map, highlight_set={a, b}),
        "edges": init_edges,
        "stats": build_stats("ni", tour_cost_val=2 * dist[a][b]),
    })

    in_tour = {a, b}
    step_idx = 2
    while len(in_tour) < n:
        chosen = -1
        chosen_d = math.inf
        for v in range(n):
            if v in in_tour:
                continue
            d_v = min(dist[v][u] for u in in_tour)
            if d_v < chosen_d:
                chosen_d = d_v
                chosen = v

        best_pos = 1
        best_delta = math.inf
        for i in range(len(tour) - 1):
            u, w = tour[i], tour[i + 1]
            delta = dist[u][chosen] + dist[chosen][w] - dist[u][w]
            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1
        u_split, w_split = tour[best_pos - 1], tour[best_pos]
        tour.insert(best_pos, chosen)
        in_tour.add(chosen)

        color_map = {nid: COLOR_NODE_VISITED for nid in in_tour}
        color_map[chosen] = COLOR_YELLOW
        edges_out = []
        for i in range(len(tour) - 1):
            u, w = tour[i], tour[i + 1]
            if (u == u_split and w == chosen) or (u == chosen and w == w_split):
                edges_out.append(edge_dict(u, w, COLOR_ORANGE, width=3,
                                           label=f"{dist[u][w]:.1f}"))
            else:
                edges_out.append(edge_dict(u, w, COLOR_PURPLE, width=2))
        cur_cost = tour_cost(tour, dist)
        steps.append({
            "title": f"Step {step_idx}: Insert node {chosen}",
            "description": (
                f"Node {chosen} is the unvisited vertex closest to the current tour "
                f"(distance {chosen_d:.2f}). Minimum insertion cost Δ = {best_delta:.2f} "
                f"occurs at edge ({u_split}, {w_split}); split that edge and insert "
                f"{chosen} (new edges in orange). Partial tour cost: {cur_cost:.2f}."
            ),
            "nodes": base_nodes(color_map=color_map, highlight_set={chosen}),
            "edges": edges_out,
            "stats": build_stats("ni", tour_cost_val=cur_cost),
        })
        step_idx += 1

    total = tour_cost(tour, dist)
    color_map = {nid: COLOR_NODE_VISITED for nid in in_tour}
    edges_out = [edge_dict(tour[i], tour[i + 1], COLOR_PURPLE, width=3,
                           label=f"{dist[tour[i]][tour[i + 1]]:.1f}")
                 for i in range(len(tour) - 1)]
    steps.append({
        "title": f"Step {step_idx}: Final tour",
        "description": (
            f"All {n} vertices inserted. Final Hamiltonian cycle cost: {total:.2f}. "
            f"Nearest Insertion guarantees ℓ(C) ≤ 2 · OPT."
        ),
        "nodes": base_nodes(color_map=color_map),
        "edges": edges_out,
        "stats": build_stats("ni", tour_cost_val=total),
    })
    return steps


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    n = int(data.get("n", 10))
    n = max(3, min(30, n))

    random.seed()
    nodes = []
    min_dist = 55
    attempts = 0
    while len(nodes) < n and attempts < 5000:
        attempts += 1
        x = random.randint(80, 720)
        y = random.randint(80, 520)
        ok = all(math.hypot(x - e["x"], y - e["y"]) >= min_dist for e in nodes)
        if ok:
            nodes.append({"id": len(nodes), "x": x, "y": y})
    while len(nodes) < n:
        nodes.append({"id": len(nodes),
                      "x": random.randint(80, 720),
                      "y": random.randint(80, 520)})

    dist = build_distance_matrix(nodes)
    state["nodes"] = nodes
    state["dist"] = dist
    state["last_steps"] = None

    # Compute optimal cost if feasible
    if n <= HELD_KARP_MAX_N:
        state["optimal_cost"] = held_karp(dist)
        state["optimal_exact"] = True
    else:
        state["optimal_cost"] = None
        state["optimal_exact"] = False

    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append({"from": i, "to": j, "dist": round(dist[i][j], 2)})

    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "optimal_cost": state["optimal_cost"],
        "optimal_exact": state["optimal_exact"],
        "n": n,
        "held_karp_max_n": HELD_KARP_MAX_N,
    })


@app.route("/run", methods=["POST"])
def run():
    if not state["nodes"]:
        return jsonify({"error": "no graph generated"}), 400
    data = request.get_json(silent=True) or {}
    algo = data.get("algorithm", "nn")

    if algo == "nn":
        steps = algo_nearest_neighbor()
    elif algo == "mst":
        steps = algo_mst_2approx()
    elif algo == "christofides":
        steps = algo_christofides()
    elif algo == "ni":
        steps = algo_nearest_insertion()
    else:
        return jsonify({"error": f"unknown algorithm {algo}"}), 400

    state["last_steps"] = steps
    return jsonify({"steps": steps, "total_steps": len(steps)})


# ---------------------------------------------------------------------------
# Lean (step-free) versions of each algorithm for benchmarking
# ---------------------------------------------------------------------------
def compute_nn(dist, n):
    visited = [False] * n
    visited[0] = True
    cur = 0
    tour = [0]
    for _ in range(n - 1):
        best = -1
        bd = math.inf
        for v in range(n):
            if not visited[v] and dist[cur][v] < bd:
                bd = dist[cur][v]
                best = v
        visited[best] = True
        tour.append(best)
        cur = best
    tour.append(0)
    return sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))


def compute_mst_tour(dist, n):
    nodes = [None] * n
    mst = prims_mst(nodes, dist)
    multi = list(mst) + list(mst)
    euler = hierholzers_eulerian(n, multi, start=0)
    ham, _ = shortcut_events(euler)
    return sum(dist[ham[i]][ham[i + 1]] for i in range(len(ham) - 1))


def compute_christofides_tour(dist, n):
    nodes = [None] * n
    mst = prims_mst(nodes, dist)
    deg = [0] * n
    for u, v in mst:
        deg[u] += 1
        deg[v] += 1
    odd = [i for i in range(n) if deg[i] % 2 == 1]
    matching = greedy_matching(odd, dist)
    multi = list(mst) + list(matching)
    euler = hierholzers_eulerian(n, multi, start=0)
    ham, _ = shortcut_events(euler)
    return sum(dist[ham[i]][ham[i + 1]] for i in range(len(ham) - 1))


def compute_ni(dist, n):
    a, b = 0, 1
    bd = math.inf
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] < bd:
                bd = dist[i][j]
                a, b = i, j
    tour = [a, b, a]
    in_tour = {a, b}
    while len(in_tour) < n:
        chosen = -1
        cd = math.inf
        for v in range(n):
            if v in in_tour:
                continue
            dv = min(dist[v][u] for u in in_tour)
            if dv < cd:
                cd = dv
                chosen = v
        bp = 1
        bdelta = math.inf
        for i in range(len(tour) - 1):
            u, w = tour[i], tour[i + 1]
            delta = dist[u][chosen] + dist[chosen][w] - dist[u][w]
            if delta < bdelta:
                bdelta = delta
                bp = i + 1
        tour.insert(bp, chosen)
        in_tour.add(chosen)
    return sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))


@app.route("/simulations")
def simulations_page():
    return render_template("simulations.html")


@app.route("/sim_round", methods=["POST"])
def sim_round():
    """Run ONE independent round: generate a random graph of n points, compute
    OPT via Held-Karp, and benchmark all four heuristics. Returns wall-clock
    runtimes (ms) and tour costs so the client can maintain running averages."""
    import time
    data = request.get_json(silent=True) or {}
    n = int(data.get("n", 10))
    n = max(4, min(HELD_KARP_MAX_N, n))

    # generate random point set with minimum spacing
    nodes = []
    min_dist = 40
    attempts = 0
    while len(nodes) < n and attempts < 4000:
        attempts += 1
        x = random.randint(80, 720)
        y = random.randint(80, 520)
        if all(math.hypot(x - e["x"], y - e["y"]) >= min_dist for e in nodes):
            nodes.append({"x": x, "y": y})
    while len(nodes) < n:
        nodes.append({"x": random.randint(80, 720),
                      "y": random.randint(80, 520)})

    dist = build_distance_matrix(nodes)

    # Optimal cost via Held-Karp
    t0 = time.perf_counter()
    opt_cost = held_karp(dist)
    opt_time = (time.perf_counter() - t0) * 1000.0

    results = {}
    for algo_name, fn in [
        ("nn", compute_nn),
        ("mst", compute_mst_tour),
        ("christofides", compute_christofides_tour),
        ("ni", compute_ni),
    ]:
        t0 = time.perf_counter()
        cost = fn(dist, n)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ratio = (cost / opt_cost) if opt_cost and opt_cost > 0 else None
        results[algo_name] = {
            "cost": round(cost, 3),
            "time_ms": round(elapsed_ms, 4),
            "ratio": round(ratio, 4) if ratio is not None else None,
        }

    return jsonify({
        "n": n,
        "optimal_cost": round(opt_cost, 3) if opt_cost is not None else None,
        "optimal_time_ms": round(opt_time, 4),
        "results": results,
    })


# ===========================================================================
# MAP PAGE: real-world Christofides on an OSRM road-network distance matrix
# ===========================================================================
OSRM_BASE = "http://router.project-osrm.org"
OSRM_TIMEOUT = 25  # seconds


def _osrm_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "tsp-lab/1.0"})
    with urllib.request.urlopen(req, timeout=OSRM_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def osrm_table(coords):
    """Call OSRM /table to get full distance (m) + duration (s) matrices.
       coords: list of (lat, lon). Returns (D_dist, D_dur)."""
    if len(coords) < 2:
        raise ValueError("need at least 2 coordinates")
    # OSRM expects lon,lat pairs separated by ;
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for (lat, lon) in coords)
    url = (f"{OSRM_BASE}/table/v1/driving/{coord_str}"
           f"?annotations=distance,duration")
    data = _osrm_get(url)
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM table error: {data.get('code')} "
                           f"{data.get('message','')}")
    return data["distances"], data["durations"]


def osrm_route(coords):
    """Call OSRM /route to get full driven geometry visiting waypoints in order.
       Returns (polyline_latlon_list, total_distance_m, total_duration_s)."""
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for (lat, lon) in coords)
    url = (f"{OSRM_BASE}/route/v1/driving/{coord_str}"
           f"?overview=full&geometries=geojson&steps=false")
    data = _osrm_get(url)
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM route error: {data.get('code')} "
                           f"{data.get('message','')}")
    route = data["routes"][0]
    # GeoJSON LineString: coords are [lon, lat] — convert to [lat, lon].
    geom = [[c[1], c[0]] for c in route["geometry"]["coordinates"]]
    return geom, route["distance"], route["duration"]


def symmetrize_min(M):
    """Return a symmetric matrix D'[i][j] = min(M[i][j], M[j][i])."""
    n = len(M)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = 0.0
            else:
                a = M[i][j]
                b = M[j][i]
                if a is None and b is None:
                    D[i][j] = float("inf")
                elif a is None:
                    D[i][j] = float(b)
                elif b is None:
                    D[i][j] = float(a)
                else:
                    D[i][j] = float(min(a, b))
    return D


def find_triangle_violations(M, eps=1e-6, max_keep=50):
    """Scan a (possibly asymmetric) matrix and return a list of triangle
    inequality violations, plus summary statistics.

    A violation is a triple (i,j,k) with d(i,k) > d(i,j) + d(j,k) + eps,
    i.e. the "direct" edge is longer than going via j.
    """
    n = len(M)
    violations = []
    worst = None
    total_triples = 0
    count = 0
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                total_triples += 1
                dik = M[i][k]
                dij = M[i][j]
                djk = M[j][k]
                if dik is None or dij is None or djk is None:
                    continue
                detour = dij + djk
                if dik > detour + eps:
                    count += 1
                    ratio = dik / detour if detour > 0 else float("inf")
                    rec = {"i": i, "j": j, "k": k,
                           "dik": round(dik, 2),
                           "via_j": round(detour, 2),
                           "ratio": round(ratio, 4)}
                    if worst is None or ratio > worst["ratio"]:
                        worst = rec
                    if len(violations) < max_keep:
                        violations.append(rec)
    return {
        "count": count,
        "total_triples": total_triples,
        "worst": worst,
        "samples": violations,
    }


def find_asymmetries(M, rel_eps=1e-3, max_keep=30):
    """List pairs (i,j) with |M[i,j]-M[j,i]| / max(M[i,j],M[j,i]) > rel_eps."""
    n = len(M)
    pairs = []
    count = 0
    worst = None
    for i in range(n):
        for j in range(i + 1, n):
            a = M[i][j]
            b = M[j][i]
            if a is None or b is None:
                continue
            mx = max(a, b)
            if mx <= 0:
                continue
            rel = abs(a - b) / mx
            if rel > rel_eps:
                count += 1
                rec = {"i": i, "j": j,
                       "forward": round(float(a), 2),
                       "backward": round(float(b), 2),
                       "rel_diff": round(rel, 4)}
                if worst is None or rel > worst["rel_diff"]:
                    worst = rec
                if len(pairs) < max_keep:
                    pairs.append(rec)
    return {"count": count, "worst": worst, "samples": pairs}


# --- step builders on a PRE-BUILT symmetric matrix (lean, no global state) ---
def _edge(u, v, color, width=3, dashed=False, label=None):
    return {"from": u, "to": v, "color": color, "width": width,
            "dashed": dashed, "label": label}


def _node_color_map(highlights=None, visited=None, odd=None):
    highlights = highlights or {}
    visited = visited or set()
    odd = odd or set()
    return {"highlights": highlights, "visited": list(visited),
            "odd": list(odd)}


def christofides_steps_on_matrix(D, coords_asym=None):
    """Build step-by-step execution of Christofides on symmetric matrix D.
    coords_asym is the ORIGINAL (non-symmetric) matrix, used only to report
    the real directional driving cost of the final Hamiltonian cycle.
    Returns (steps, result_dict).
    """
    n = len(D)
    steps = []

    steps.append({
        "title": "Step 1 — Setup",
        "description": (
            "The distance matrix obtained from OSRM is asymmetric (one-way "
            "streets etc.) and may violate the triangle inequality. We run "
            "Christofides on the SYMMETRIZED matrix D′[i,j] = min(d(i,j), "
            "d(j,i)); once we obtain a vertex ordering we will re-price it on "
            "the real directional matrix and compare."
        ),
        "highlight_nodes": [],
        "edges": [],
    })

    # MST
    mst = prims_mst([None] * n, D)
    mst_cost = sum(D[u][v] for (u, v) in mst)
    steps.append({
        "title": "Step 2 — Minimum spanning tree",
        "description": (
            f"Prim's algorithm on D′ gives an MST T of weight ℓ(T) = "
            f"{mst_cost:.2f} m. T is a lower bound on the symmetrized TSP "
            f"optimum."
        ),
        "highlight_nodes": [],
        "edges": [_edge(u, v, COLOR_BLUE, 4, label=f"{D[u][v]:.0f}")
                  for (u, v) in mst],
    })

    # Odd-degree vertices
    deg = [0] * n
    for u, v in mst:
        deg[u] += 1
        deg[v] += 1
    odd = [i for i in range(n) if deg[i] % 2 == 1]
    steps.append({
        "title": "Step 3 — Odd-degree vertices in T",
        "description": (
            f"{len(odd)} vertices have odd degree in T (always even count, "
            f"by the Handshaking Lemma). We will add a matching on these."
        ),
        "highlight_nodes": list(odd),
        "edges": [_edge(u, v, COLOR_BLUE, 4) for (u, v) in mst],
    })

    # Matching
    matching = greedy_matching(odd, D)
    match_cost = sum(D[u][v] for (u, v) in matching)
    steps.append({
        "title": "Step 4 — Greedy perfect matching on odd vertices",
        "description": (
            f"Weight of matching M = {match_cost:.2f} m. Theoretical "
            f"minimum-weight matching would satisfy ℓ(M*) ≤ ½·OPT on a metric "
            f"instance; on a non-metric instance this bound need not hold."
        ),
        "highlight_nodes": list(odd),
        "edges": ([_edge(u, v, COLOR_BLUE, 4) for (u, v) in mst] +
                  [_edge(u, v, COLOR_GREEN, 4, dashed=True,
                         label=f"{D[u][v]:.0f}") for (u, v) in matching]),
    })

    # Eulerian
    multi_edges = list(mst) + list(matching)
    euler = hierholzers_eulerian(n, multi_edges, start=0)
    steps.append({
        "title": "Step 5 — Eulerian circuit of T ∪ M",
        "description": (
            f"Hierholzer's algorithm traverses each edge of T ∪ M exactly "
            f"once. The Eulerian circuit has length ℓ(T) + ℓ(M) = "
            f"{mst_cost + match_cost:.2f} m and visits {len(euler) - 1} edges."
        ),
        "highlight_nodes": [],
        "edges": [_edge(euler[i], euler[i + 1], COLOR_ORANGE, 3,
                        label=str(i + 1))
                  for i in range(len(euler) - 1)],
    })

    # Shortcut — ONE step per event, showing replaced subpath in red + new edge in yellow
    ham, events = shortcut_events(euler)
    committed = []
    for ev in events:
        u, v, sub = ev["from"], ev["to"], ev["subpath"]
        sub_edges = [(sub[i], sub[i + 1]) for i in range(len(sub) - 1)]
        replaced_cost = sum(D[a][b] for (a, b) in sub_edges)
        direct = D[u][v]
        note = ""
        if direct > replaced_cost + 1e-6:
            note = (f" ⚠ triangle inequality VIOLATED: d({u},{v}) = "
                    f"{direct:.1f} > Euler subpath {replaced_cost:.1f} — the "
                    f"shortcut actually lengthens the tour.")
        desc = (
            f"From node {u}, the Eulerian circuit passes through "
            f"{', '.join(str(s) for s in ev['skipped']) if ev['skipped'] else '—'} "
            f"before reaching next unvisited node {v}. Replace that subpath "
            f"(red, cost {replaced_cost:.1f}) by the direct edge "
            f"({u},{v}) of cost {direct:.1f} (yellow).{note}"
        )
        edges_vis = []
        for (a, b) in committed:
            edges_vis.append(_edge(a, b, COLOR_PURPLE, 4))
        for (a, b) in sub_edges:
            edges_vis.append(_edge(a, b, COLOR_RED, 3, dashed=True,
                                   label=f"{D[a][b]:.0f}"))
        edges_vis.append(_edge(u, v, COLOR_YELLOW, 5,
                               label=f"{direct:.0f}"))
        steps.append({
            "title": f"Step — Shortcut → node {v}"
                     + (" (close cycle)" if ev["closing"] else ""),
            "description": desc,
            "highlight_nodes": [u, v],
            "edges": edges_vis,
        })
        committed.append((u, v))

    ham_cost_sym = sum(D[ham[i]][ham[i + 1]] for i in range(len(ham) - 1))
    # Also price on original asymmetric matrix (real driving cost)
    real_cost = None
    if coords_asym is not None:
        try:
            real_cost = sum(float(coords_asym[ham[i]][ham[i + 1]])
                            for i in range(len(ham) - 1))
        except Exception:
            real_cost = None

    extra = ""
    if real_cost is not None and abs(real_cost - ham_cost_sym) > 1e-3:
        extra = (f" Priced on the real directional matrix, the same ordering "
                 f"costs {real_cost:.2f} m — the driver will actually cover "
                 f"that distance.")
    final_edges = [_edge(ham[i], ham[i + 1], COLOR_PURPLE, 4,
                         label=f"{D[ham[i]][ham[i + 1]]:.0f}")
                   for i in range(len(ham) - 1)]
    steps.append({
        "title": "Final Hamiltonian cycle",
        "description": (
            f"Hamiltonian cycle length on the symmetrized matrix: "
            f"ℓ(C) = {ham_cost_sym:.2f} m.{extra}"
        ),
        "highlight_nodes": [],
        "edges": final_edges,
    })

    return steps, {
        "mst": mst, "mst_cost": mst_cost,
        "odd": odd,
        "matching": matching, "matching_cost": match_cost,
        "euler": euler,
        "hamiltonian": ham,
        "ham_cost_sym": ham_cost_sym,
        "ham_cost_real": real_cost,
    }


@app.route("/map")
def map_page():
    return render_template("map.html")


@app.route("/map_distances", methods=["POST"])
def map_distances():
    data = request.get_json(silent=True) or {}
    points = data.get("points") or []
    if len(points) < 2:
        return jsonify({"error": "need at least 2 points"}), 400
    if len(points) > 20:
        return jsonify({"error": "at most 20 points"}), 400
    coords = [(float(p["lat"]), float(p["lng"])) for p in points]
    try:
        dist, dur = osrm_table(coords)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return jsonify({"error": f"OSRM request failed: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"distances": dist, "durations": dur, "n": len(coords)})


@app.route("/map_route", methods=["POST"])
def map_route():
    """Return the real road geometry between given waypoints (in order)."""
    data = request.get_json(silent=True) or {}
    points = data.get("points") or []
    if len(points) < 2:
        return jsonify({"error": "need at least 2 points"}), 400
    coords = [(float(p["lat"]), float(p["lng"])) for p in points]
    try:
        geom, dist_m, dur_s = osrm_route(coords)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return jsonify({"error": f"OSRM request failed: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({
        "polyline": geom, "distance_m": dist_m, "duration_s": dur_s,
    })


@app.route("/map_solve", methods=["POST"])
def map_solve():
    """Given points + the (asymmetric) distance matrix, run Christofides on
    the symmetrized matrix and return step-by-step execution, statistics,
    triangle-inequality violations, and the final Hamiltonian ordering."""
    data = request.get_json(silent=True) or {}
    points = data.get("points") or []
    M = data.get("matrix")  # asymmetric distance matrix (meters)
    if not points or not M or len(points) < 3:
        return jsonify({"error": "need at least 3 points and a matrix"}), 400
    n = len(points)
    if len(M) != n or any(len(row) != n for row in M):
        return jsonify({"error": "matrix shape mismatch"}), 400

    D = symmetrize_min(M)
    # Quick sanity: reject if any off-diagonal is inf (OSRM could not route)
    for i in range(n):
        for j in range(n):
            if i != j and (D[i][j] == float("inf")):
                return jsonify({
                    "error": f"no route between points {i} and {j}"
                }), 400

    triangle = find_triangle_violations(M)
    asym = find_asymmetries(M)

    steps, result = christofides_steps_on_matrix(D, coords_asym=M)

    # Per-leg real driving costs along the Hamiltonian ordering
    ham = result["hamiltonian"]
    legs = []
    for i in range(len(ham) - 1):
        a, b = ham[i], ham[i + 1]
        legs.append({
            "from": a, "to": b,
            "sym_m": round(float(D[a][b]), 2),
            "real_m": round(float(M[a][b]), 2) if M[a][b] is not None else None,
        })

    return jsonify({
        "steps": steps,
        "hamiltonian": ham,
        "mst_edges": result["mst"],
        "matching_edges": result["matching"],
        "odd_vertices": result["odd"],
        "euler": result["euler"],
        "stats": {
            "n": n,
            "mst_cost_m": round(result["mst_cost"], 2),
            "matching_cost_m": round(result["matching_cost"], 2),
            "ham_cost_sym_m": round(result["ham_cost_sym"], 2),
            "ham_cost_real_m": (round(result["ham_cost_real"], 2)
                                if result["ham_cost_real"] is not None else None),
        },
        "legs": legs,
        "triangle_violations": triangle,
        "asymmetries": asym,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234, debug=True)
