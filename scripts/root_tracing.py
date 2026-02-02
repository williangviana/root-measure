import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.graph import route_through_array

from config import (SCALE_PX_PER_CM, min_component_size,
                    roi_half_width_px, roi_vertical_px, roi_pad_px,
                    max_click_distance_px)

_SQRT2 = np.sqrt(2)


# ---------------------------------------------------------------------------
# Sparse graph utilities
# ---------------------------------------------------------------------------

def _build_sparse_graph(skeleton):
    """Build a sparse weighted adjacency matrix from a skeleton image.

    Returns (skel_points, graph, idx_image) where:
      skel_points : Nx2 int array of (row, col) coords
      graph       : csr_matrix (N, N) weighted adjacency
      idx_image   : 2-D int32 array same shape as skeleton, pixel -> node index or -1
    """
    skel_points = np.argwhere(skeleton)
    N = len(skel_points)
    idx_image = np.full(skeleton.shape, -1, dtype=np.int32)
    if N == 0:
        return skel_points, csr_matrix((0, 0)), idx_image

    idx_image[skel_points[:, 0], skel_points[:, 1]] = np.arange(N, dtype=np.int32)

    h, w = skeleton.shape
    rs = skel_points[:, 0]
    cs = skel_points[:, 1]

    offsets = [(-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
               ( 0, -1, 1.0),                   ( 0, 1, 1.0),
               ( 1, -1, _SQRT2), ( 1, 0, 1.0),  ( 1, 1, _SQRT2)]

    rows_i = []
    cols_j = []
    weights = []

    for dr, dc, dist in offsets:
        nr = rs + dr
        nc = cs + dc
        valid = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        vi = np.where(valid)[0]
        ni = idx_image[nr[vi], nc[vi]]
        has = ni >= 0
        src = vi[has]
        dst = ni[has]
        rows_i.append(src)
        cols_j.append(dst)
        weights.append(np.full(len(src), dist))

    rows_i = np.concatenate(rows_i)
    cols_j = np.concatenate(cols_j)
    weights = np.concatenate(weights)

    graph = csr_matrix((weights, (rows_i, cols_j)), shape=(N, N))
    return skel_points, graph, idx_image


def _path_from_predecessors(predecessors, start, end):
    """Reconstruct node-index path from scipy dijkstra predecessors."""
    if start == end:
        return [start]
    path = []
    node = end
    while node != start:
        if node < 0:
            return None
        path.append(node)
        node = predecessors[node]
    path.append(start)
    path.reverse()
    return path


def _snap_sparse(kdtree, point, comp_labels, comp_sizes, min_size,
                 scale=SCALE_PX_PER_CM, k=200):
    """Snap a point to the nearest skeleton node in a large component."""
    k = min(k, kdtree.n)
    if k == 0:
        return None
    dists, indices = kdtree.query(point, k=k)
    if k == 1:
        dists = [dists]
        indices = [indices]
    max_dist = max_click_distance_px(scale)
    for d, i in zip(dists, indices):
        if d > max_dist:
            return None
        i = int(i)
        if comp_sizes[comp_labels[i]] >= min_size:
            return i
    return None


# ---------------------------------------------------------------------------
# Plate-level pre-computation
# ---------------------------------------------------------------------------

def build_plate_graph(binary_image, plate_bounds, scale=SCALE_PX_PER_CM):
    """Skeletonize a plate region once and build a reusable sparse graph.

    Parameters
    ----------
    binary_image : 2-D bool array (full image)
    plate_bounds : (r1, r2, c1, c2) plate region
    scale : px/cm

    Returns dict with skeleton, graph, kdtree, components, degrees.
    """
    r1, r2, c1, c2 = plate_bounds
    roi = binary_image[r1:r2, c1:c2]
    skeleton = skeletonize(roi)
    skel_points, graph, idx_image = _build_sparse_graph(skeleton)
    N = len(skel_points)

    if N == 0:
        return dict(skeleton=skeleton, skel_points=skel_points,
                    graph=graph, idx_image=idx_image,
                    comp_labels=np.empty(0, dtype=np.int32),
                    comp_sizes=np.empty(0, dtype=np.int64),
                    degrees=np.empty(0, dtype=np.int32),
                    kdtree=None, offset=(r1, c1), bounds=plate_bounds)

    n_comp, comp_labels = connected_components(graph, directed=False)
    comp_sizes = np.bincount(comp_labels)
    degrees = np.diff(graph.indptr)
    kdtree = cKDTree(skel_points)

    return dict(skeleton=skeleton, skel_points=skel_points,
                graph=graph, idx_image=idx_image,
                comp_labels=comp_labels, comp_sizes=comp_sizes,
                degrees=degrees, kdtree=kdtree,
                offset=(r1, c1), bounds=plate_bounds)


# ---------------------------------------------------------------------------
# Tip finding
# ---------------------------------------------------------------------------

def find_root_tip(binary_image, top_point, scale=SCALE_PX_PER_CM,
                  plate_bounds=None, plate_graph=None):
    """Find the root tip (bottom endpoint) starting from a top click.

    Always uses a local ROI for tip detection to isolate each root
    from neighbours.  The plate_graph is only used for tracing.

    Returns (tip_row, tip_col) in full image coordinates, or None.
    """
    return _find_root_tip_local(binary_image, top_point, scale, plate_bounds)


def _find_root_tip_fast(top_point, pg, scale):
    """Tip finding using a prebuilt plate graph.

    Uses graph distance to stay on the same root rather than jumping
    to a neighbouring root's endpoint that happens to be further down.
    """
    sp = pg['skel_points']
    if len(sp) == 0:
        return None
    off_r, off_c = pg['offset']
    top_local = (top_point[0] - off_r, top_point[1] - off_c)

    min_comp = min_component_size(scale)
    start_idx = _snap_sparse(pg['kdtree'], top_local, pg['comp_labels'],
                             pg['comp_sizes'], min_comp, scale)
    if start_idx is None:
        return None

    comp_label = pg['comp_labels'][start_idx]
    comp_mask = pg['comp_labels'] == comp_label
    endpoints = np.where(comp_mask & (pg['degrees'] == 1))[0]
    start_row = sp[start_idx][0]

    # always compute graph distances — needed for scoring
    graph_dists = dijkstra(pg['graph'], indices=start_idx)

    if len(endpoints) == 0:
        graph_dists[~comp_mask] = -1
        above = sp[:, 0] < start_row
        graph_dists[above] = -1
        furthest = int(np.argmax(graph_dists))
        tip_local = sp[furthest]
    else:
        below = endpoints[sp[endpoints, 0] >= start_row]

        if len(below) > 0:
            # Prefer endpoints whose graph distance is close to their
            # Euclidean distance from the start — i.e. a direct path,
            # not one that detours through neighbouring roots.
            start_col = sp[start_idx][1]
            drop = (sp[below, 0] - start_row).astype(float)
            lateral = np.abs(sp[below, 1] - start_col).astype(float)
            eucl = np.sqrt(drop ** 2 + lateral ** 2)
            eucl = np.where(eucl < 1.0, 1.0, eucl)
            gdist = graph_dists[below]
            gdist = np.where(gdist <= 0, 1.0, gdist)
            # directness: 1.0 = path follows skeleton tightly, <1 = detour
            directness = eucl / gdist
            # score: vertical drop weighted by how direct the path is
            scores = drop * directness
            tip_idx = below[np.argmax(scores)]
        else:
            graph_dists[sp[:, 0] < start_row] = np.inf
            valid = graph_dists[endpoints]
            if np.all(np.isinf(valid)):
                # truly unreachable — pick furthest by graph distance
                all_dists = dijkstra(pg['graph'], indices=start_idx)
                tip_idx = endpoints[np.argmax(all_dists[endpoints])]
            else:
                tip_idx = endpoints[np.argmin(valid)]
        tip_local = sp[tip_idx]

    return (int(tip_local[0] + off_r), int(tip_local[1] + off_c))


def _find_root_tip_local(binary_image, top_point, scale, plate_bounds):
    """Tip finding with local skeletonization (fallback path)."""
    h, w = binary_image.shape
    if not (0 <= top_point[0] < h and 0 <= top_point[1] < w):
        return None

    half_w = roi_half_width_px(scale)
    vert = roi_vertical_px(scale)

    rmin = max(0, top_point[0] - 10)
    rmax = min(h, top_point[0] + vert)
    cmin = max(0, top_point[1] - half_w)
    cmax = min(w, top_point[1] + half_w)

    if plate_bounds is not None:
        pr1, pr2, pc1, pc2 = plate_bounds
        rmin = max(rmin, pr1)
        rmax = min(rmax, pr2)
        cmin = max(cmin, pc1)
        cmax = min(cmax, pc2)

    roi = binary_image[rmin:rmax, cmin:cmax]
    top_local = (top_point[0] - rmin, top_point[1] - cmin)

    skeleton = skeletonize(roi)
    if skeleton.sum() == 0:
        return None

    skel_points, graph, idx_image = _build_sparse_graph(skeleton)
    N = len(skel_points)
    if N == 0:
        return None

    n_comp, comp_labels = connected_components(graph, directed=False)
    comp_sizes = np.bincount(comp_labels)
    degrees = np.diff(graph.indptr)
    kdtree = cKDTree(skel_points)

    min_comp = min_component_size(scale)
    start_idx = _snap_sparse(kdtree, top_local, comp_labels, comp_sizes,
                             min_comp, scale)
    if start_idx is None:
        return None

    comp_label = comp_labels[start_idx]
    comp_mask = comp_labels == comp_label
    endpoints = np.where(comp_mask & (degrees == 1))[0]
    start_row = skel_points[start_idx][0]

    if len(endpoints) == 0:
        dists = dijkstra(graph, indices=start_idx)
        dists[~comp_mask] = -1
        above = skel_points[:, 0] < start_row
        dists[above] = -1
        furthest = int(np.argmax(dists))
        tip_local = skel_points[furthest]
    else:
        below = endpoints[skel_points[endpoints, 0] >= start_row]

        if len(below) > 0:
            start_col = skel_points[start_idx][1]
            scores = (skel_points[below, 0]
                      - 2.0 * np.abs(skel_points[below, 1] - start_col))
            tip_idx = below[np.argmax(scores)]
        else:
            dists = dijkstra(graph, indices=start_idx)
            dists[skel_points[:, 0] < start_row] = np.inf
            valid = dists[endpoints]
            if np.all(np.isinf(valid)):
                tip_idx = endpoints[np.argmax(
                    dijkstra(graph, indices=start_idx)[endpoints])]
            else:
                tip_idx = endpoints[np.argmax(valid)]
        tip_local = skel_points[tip_idx]

    return (int(tip_local[0] + rmin), int(tip_local[1] + cmin))


# ---------------------------------------------------------------------------
# Root tracing
# ---------------------------------------------------------------------------

def trace_root(binary_image, top_point, bottom_point, scale=SCALE_PX_PER_CM,
               plate_bounds=None, plate_graph=None):
    """Trace a single root between two clicked points.

    Returns dict with length_cm, length_px, path, method, warning.
    """
    h, w = binary_image.shape
    for label, pt in [('Top', top_point), ('Bottom', bottom_point)]:
        if not (0 <= pt[0] < h and 0 <= pt[1] < w):
            return dict(length_cm=0, length_px=0, path=np.empty((0, 2)),
                        method='error', warning=f'{label} point outside image')

    # try Phase 1 on prebuilt graph
    path_full = None
    method = 'skeleton_graph'
    if plate_graph is not None:
        path_full = _try_graph_sparse(plate_graph, top_point, bottom_point, scale)

    # Phase 1 fallback: build local graph
    if path_full is None and plate_graph is None:
        path_full, method = _trace_local_phase1(
            binary_image, top_point, bottom_point, scale, plate_bounds)

    # Phase 2 fallback: hybrid cost path
    if path_full is None:
        path_full, method = _trace_phase2(
            binary_image, top_point, bottom_point, scale, plate_bounds)

    if path_full is None or len(path_full) == 0:
        return dict(length_cm=0, length_px=0, path=np.empty((0, 2)),
                    method='error', warning='No root skeleton found near clicks')

    diffs = np.diff(path_full.astype(np.float64), axis=0)
    length_px = np.sqrt((diffs ** 2).sum(axis=1)).sum()
    length_cm = length_px / scale

    return dict(length_cm=length_cm, length_px=length_px, path=path_full,
                method=method, warning=None)


def _try_graph_sparse(pg, top_point, bottom_point, scale):
    """Phase 1 on prebuilt plate graph. Returns path in image coords or None."""
    sp = pg['skel_points']
    if len(sp) == 0:
        return None
    off_r, off_c = pg['offset']
    top_local = (top_point[0] - off_r, top_point[1] - off_c)
    bot_local = (bottom_point[0] - off_r, bottom_point[1] - off_c)

    min_comp = min_component_size(scale)
    idx_start = _snap_sparse(pg['kdtree'], top_local, pg['comp_labels'],
                             pg['comp_sizes'], min_comp, scale)
    idx_end = _snap_sparse(pg['kdtree'], bot_local, pg['comp_labels'],
                           pg['comp_sizes'], min_comp, scale)
    if idx_start is None or idx_end is None:
        return None
    if pg['comp_labels'][idx_start] != pg['comp_labels'][idx_end]:
        return None

    dists, preds = dijkstra(pg['graph'], indices=idx_start,
                            return_predecessors=True)
    path_idx = _path_from_predecessors(preds, idx_start, idx_end)
    if path_idx is None:
        return None
    path_local = sp[path_idx]
    return path_local + np.array([off_r, off_c])


def _trace_local_phase1(binary_image, top_point, bottom_point,
                        scale, plate_bounds):
    """Phase 1 with local skeletonization. Returns (path_full, method) or (None, _)."""
    h, w = binary_image.shape
    pad = roi_pad_px(scale)
    half_w = roi_half_width_px(scale)
    rmin = max(0, min(top_point[0], bottom_point[0]) - pad)
    rmax = min(h, max(top_point[0], bottom_point[0]) + pad)
    cmin = max(0, min(top_point[1], bottom_point[1]) - half_w)
    cmax = min(w, max(top_point[1], bottom_point[1]) + half_w)

    if plate_bounds is not None:
        pr1, pr2, pc1, pc2 = plate_bounds
        rmin = max(rmin, pr1)
        rmax = min(rmax, pr2)
        cmin = max(cmin, pc1)
        cmax = min(cmax, pc2)

    roi = binary_image[rmin:rmax, cmin:cmax]
    skeleton = skeletonize(roi)
    if skeleton.sum() == 0:
        return None, 'error'

    skel_points, graph, idx_image = _build_sparse_graph(skeleton)
    if len(skel_points) == 0:
        return None, 'error'

    n_comp, comp_labels = connected_components(graph, directed=False)
    comp_sizes = np.bincount(comp_labels)
    kdtree = cKDTree(skel_points)

    top_local = (top_point[0] - rmin, top_point[1] - cmin)
    bot_local = (bottom_point[0] - rmin, bottom_point[1] - cmin)
    min_comp = min_component_size(scale)

    idx_start = _snap_sparse(kdtree, top_local, comp_labels, comp_sizes,
                             min_comp, scale)
    idx_end = _snap_sparse(kdtree, bot_local, comp_labels, comp_sizes,
                           min_comp, scale)
    if idx_start is None or idx_end is None:
        return None, 'error'
    if comp_labels[idx_start] != comp_labels[idx_end]:
        return None, 'error'

    dists, preds = dijkstra(graph, indices=idx_start, return_predecessors=True)
    path_idx = _path_from_predecessors(preds, idx_start, idx_end)
    if path_idx is None:
        return None, 'error'

    path_full = skel_points[path_idx] + np.array([rmin, cmin])
    return path_full, 'skeleton_graph'


def _trace_phase2(binary_image, top_point, bottom_point, scale, plate_bounds):
    """Phase 2 fallback: hybrid cost path. Returns (path_full, method)."""
    h, w = binary_image.shape
    pad = roi_pad_px(scale)
    half_w = roi_half_width_px(scale)
    rmin = max(0, min(top_point[0], bottom_point[0]) - pad)
    rmax = min(h, max(top_point[0], bottom_point[0]) + pad)
    cmin = max(0, min(top_point[1], bottom_point[1]) - half_w)
    cmax = min(w, max(top_point[1], bottom_point[1]) + half_w)

    if plate_bounds is not None:
        pr1, pr2, pc1, pc2 = plate_bounds
        rmin = max(rmin, pr1)
        rmax = min(rmax, pr2)
        cmin = max(cmin, pc1)
        cmax = min(cmax, pc2)

    roi = binary_image[rmin:rmax, cmin:cmax]
    skeleton = skeletonize(roi)
    if skeleton.sum() == 0:
        return None, 'error'

    top_local = (top_point[0] - rmin, top_point[1] - cmin)
    bot_local = (bottom_point[0] - rmin, bottom_point[1] - cmin)

    path_local = _hybrid_cost_path(roi, skeleton, top_local, bot_local)
    path_full = path_local + np.array([rmin, cmin])
    return path_full, 'hybrid_cost'


def _hybrid_cost_path(binary_roi, skeleton, start, end):
    """Phase 2 fallback: cost-map path preferring skeleton > root > background."""
    cost = np.full(binary_roi.shape, 10000.0, dtype=np.float64)
    cost[binary_roi] = 5.0
    cost[skeleton] = 1.0

    path_list, _ = route_through_array(cost, start, end, fully_connected=True)
    return np.array(path_list)
