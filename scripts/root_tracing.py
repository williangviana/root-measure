import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.graph import route_through_array

from config import (SCALE_PX_PER_CM, min_component_size,
                    roi_half_width_px, roi_vertical_px, roi_pad_px,
                    max_click_distance_px)


def find_root_tip(binary_image, top_point, scale=SCALE_PX_PER_CM):
    """Find the root tip (bottom endpoint) starting from a top click.

    Traces the skeleton downward from the starting point to find the
    furthest endpoint (the root tip).

    Returns (tip_row, tip_col) in full image coordinates, or None if not found.
    """
    h, w = binary_image.shape

    # validate bounds
    if not (0 <= top_point[0] < h and 0 <= top_point[1] < w):
        return None

    # compute DPI-dynamic ROI limits
    half_w = roi_half_width_px(scale)
    vert = roi_vertical_px(scale)

    # extract ROI starting at the click point — downward only
    # no pixels above the click, so tracing can only go down
    rmin = max(0, top_point[0] - 10)    # tiny margin for click imprecision
    rmax = min(h, top_point[0] + vert)  # roots up to ~10 cm (plate limit)
    cmin = max(0, top_point[1] - half_w)
    cmax = min(w, top_point[1] + half_w)

    roi = binary_image[rmin:rmax, cmin:cmax]
    top_local = (top_point[0] - rmin, top_point[1] - cmin)

    # skeletonize the ROI
    skeleton = skeletonize(roi)
    if skeleton.sum() == 0:
        return None

    skel_points = np.argwhere(skeleton)
    if len(skel_points) == 0:
        return None

    # build graph
    coord_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(skel_points)}

    edges = []
    for i, (r, c) in enumerate(skel_points):
        r, c = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = coord_to_idx.get((r + dr, c + dc))
                if j is not None and j > i:
                    dist = np.sqrt(2) if (dr != 0 and dc != 0) else 1.0
                    edges.append((i, j, dist))

    G = nx.Graph()
    G.add_nodes_from(range(len(skel_points)))
    G.add_weighted_edges_from(edges)

    # find component sizes
    node_comp_size = {}
    for comp in nx.connected_components(G):
        sz = len(comp)
        for n in comp:
            node_comp_size[n] = sz

    # snap click to nearest skeleton point in a large component
    tree = cKDTree(skel_points)
    dist, idx = tree.query(top_local, k=min(100, len(skel_points)))
    if np.isscalar(dist):
        dist, idx = [dist], [idx]

    start_idx = None
    max_dist = max_click_distance_px(scale)
    for d, i in zip(dist, idx):
        if d > max_dist:
            break
        if node_comp_size.get(int(i), 0) >= min_component_size(scale):
            start_idx = int(i)
            break

    if start_idx is None:
        return None

    # find all endpoints (degree 1 nodes) in the same component
    component = nx.node_connected_component(G, start_idx)
    subG = G.subgraph(component).copy()

    # prune short lateral branches before tip selection
    _prune_lateral_branches(subG, {start_idx}, scale)

    # find endpoints in pruned graph
    endpoints = [n for n in subG.nodes() if subG.degree(n) == 1]

    if not endpoints:
        # no clear endpoints — use the furthest point from start
        lengths = nx.single_source_dijkstra_path_length(subG, start_idx, weight='weight')
        furthest = max(subG.nodes(), key=lambda n: lengths.get(n, 0))
        tip_local = skel_points[furthest]
    else:
        # find the endpoint furthest down (max row) that's reachable from start
        # prefer endpoints that are below the start point
        start_row = skel_points[start_idx][0]
        below_endpoints = [n for n in endpoints if skel_points[n][0] > start_row]

        if below_endpoints:
            # pick the one furthest down, penalize lateral drift
            # to avoid jumping to neighboring roots
            start_col = skel_points[start_idx][1]
            def _tip_score(n):
                r, c = skel_points[n]
                return r - 2.0 * abs(c - start_col)
            tip_idx = max(below_endpoints, key=_tip_score)
        else:
            # fallback: furthest endpoint by path length
            lengths = nx.single_source_dijkstra_path_length(subG, start_idx, weight='weight')
            tip_idx = max(endpoints, key=lambda n: lengths.get(n, 0))

        tip_local = skel_points[tip_idx]

    # convert back to full image coordinates
    tip_full = (tip_local[0] + rmin, tip_local[1] + cmin)
    return tip_full


def _prune_lateral_branches(G, protected_nodes, scale):
    """Remove short lateral branches from skeleton graph.

    Iteratively removes endpoints whose branch (path to nearest junction)
    is shorter than the prune threshold. Never removes protected nodes.
    Modifies G in place.
    """
    prune_thresh = roi_half_width_px(scale) * 0.4
    changed = True
    while changed:
        changed = False
        endpoints = [n for n in G.nodes() if G.degree(n) == 1
                     and n not in protected_nodes]
        for ep in endpoints:
            if ep not in G:
                continue
            branch_len = 0.0
            cur = ep
            prev = None
            while True:
                neighbors = [nb for nb in G.neighbors(cur) if nb != prev]
                if len(neighbors) != 1:
                    break
                nxt = neighbors[0]
                branch_len += G[cur][nxt].get('weight', 1.0)
                if branch_len > prune_thresh:
                    break
                prev = cur
                cur = nxt
            if branch_len <= prune_thresh:
                cur = ep
                prev = None
                while True:
                    neighbors = [nb for nb in G.neighbors(cur) if nb != prev]
                    nxt = neighbors[0] if len(neighbors) == 1 else None
                    G.remove_node(cur)
                    if nxt is None or G.degree(nxt) != 1 or nxt in protected_nodes:
                        break
                    prev = cur
                    cur = nxt
                changed = True


def _snap_to_large_component(tree, point, node_comp_size, min_size,
                             scale=SCALE_PX_PER_CM, k=200):
    """Find nearest skeleton pixel belonging to a component >= min_size."""
    k = min(k, tree.n)
    dists, indices = tree.query(point, k=k)
    if k == 1:
        dists = [dists]
        indices = [indices]
    max_dist = max_click_distance_px(scale)
    for d, i in zip(dists, indices):
        if d > max_dist:
            return None
        if node_comp_size.get(int(i), 0) >= min_size:
            return int(i)
    return None


def _try_skeleton_graph(skeleton, start, end, scale=SCALE_PX_PER_CM):
    """Phase 1: shortest path on skeleton graph. Returns path array or None."""
    skel_points = np.argwhere(skeleton)
    if len(skel_points) == 0:
        return None

    coord_to_idx = {}
    for i, (r, c) in enumerate(skel_points):
        coord_to_idx[(int(r), int(c))] = i

    edges = []
    for i, (r, c) in enumerate(skel_points):
        r, c = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = coord_to_idx.get((r + dr, c + dc))
                if j is not None and j > i:
                    dist = np.sqrt(2) if (dr != 0 and dc != 0) else 1.0
                    edges.append((i, j, dist))

    G = nx.Graph()
    G.add_nodes_from(range(len(skel_points)))
    G.add_weighted_edges_from(edges)

    # component sizes
    node_comp_size = {}
    for comp in nx.connected_components(G):
        sz = len(comp)
        for n in comp:
            node_comp_size[n] = sz

    tree = cKDTree(skel_points)

    min_comp = min_component_size(scale)
    idx_start = _snap_to_large_component(tree, start, node_comp_size,
                                         min_comp, scale=scale)
    idx_end = _snap_to_large_component(tree, end, node_comp_size,
                                       min_comp, scale=scale)

    if idx_start is None or idx_end is None:
        return None
    if not nx.has_path(G, idx_start, idx_end):
        return None

    # prune lateral branches so the path stays on the primary root
    _prune_lateral_branches(G, {idx_start, idx_end}, scale)

    if not nx.has_path(G, idx_start, idx_end):
        return None

    path_idx = nx.dijkstra_path(G, idx_start, idx_end, weight='weight')
    return skel_points[path_idx]


def _hybrid_cost_path(binary_roi, skeleton, start, end):
    """Phase 2 fallback: cost-map path preferring skeleton > root > background."""
    cost = np.full(binary_roi.shape, 10000.0, dtype=np.float64)
    cost[binary_roi] = 5.0
    cost[skeleton] = 1.0

    path_list, _ = route_through_array(cost, start, end, fully_connected=True)
    return np.array(path_list)


def trace_root(binary_image, top_point, bottom_point, scale=SCALE_PX_PER_CM):
    """Trace a single root between two clicked points.

    Returns dict with length_cm, length_px, path, method, warning.
    """
    h, w = binary_image.shape

    # validate bounds
    for label, pt in [('Top', top_point), ('Bottom', bottom_point)]:
        if not (0 <= pt[0] < h and 0 <= pt[1] < w):
            return dict(length_cm=0, length_px=0, path=np.empty((0, 2)),
                        method='error', warning=f'{label} point outside image')

    # extract local ROI (DPI-dynamic)
    pad = roi_pad_px(scale)
    half_w = roi_half_width_px(scale)
    rmin = max(0, min(top_point[0], bottom_point[0]) - pad)
    rmax = min(h, max(top_point[0], bottom_point[0]) + pad)
    cmin = max(0, min(top_point[1], bottom_point[1]) - half_w)
    cmax = min(w, max(top_point[1], bottom_point[1]) + half_w)

    roi = binary_image[rmin:rmax, cmin:cmax]
    top_local = (top_point[0] - rmin, top_point[1] - cmin)
    bot_local = (bottom_point[0] - rmin, bottom_point[1] - cmin)

    # skeletonize the ROI
    skeleton = skeletonize(roi)

    if skeleton.sum() == 0:
        return dict(length_cm=0, length_px=0, path=np.empty((0, 2)),
                    method='error', warning='No root skeleton found near clicks')

    # Phase 1: skeleton graph
    path_local = _try_skeleton_graph(skeleton, top_local, bot_local, scale=scale)
    method = 'skeleton_graph'

    # Phase 2: fallback
    if path_local is None:
        path_local = _hybrid_cost_path(roi, skeleton, top_local, bot_local)
        method = 'hybrid_cost'

    # convert to full-image coords
    path_full = path_local + np.array([rmin, cmin])

    # path length
    diffs = np.diff(path_full.astype(np.float64), axis=0)
    length_px = np.sqrt((diffs ** 2).sum(axis=1)).sum()
    length_cm = length_px / scale

    return dict(length_cm=length_cm, length_px=length_px, path=path_full,
                method=method, warning=None)
