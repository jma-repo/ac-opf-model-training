"""Match CODERS node-to-node branches to Government of Alberta powerline
geometry.

Method (derived:goa_path_match):
 1. Explode GoA polylines (EPSG:3400, metres) into vertices / segments.
 2. Stitch: a polyline endpoint within `stitch_tol_m` of any other vertex is
    joined to it (union-find). Crossings without a shared vertex stay apart.
 3. Snap every CODERS node to its nearest GoA vertex (<= substation_snap_m).
 4. For each branch, shortest path on the GoA graph between the snapped
    endpoints. Accept if path_len / route_len is within `match_ratio`.
Unmatched branches get a straight-line geometry (synthetic:straight_line).
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString

log = logging.getLogger("abgrid.conflate.goa")


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


class GoAGraph:
    def __init__(self, goa: gpd.GeoDataFrame, stitch_tol_m: float):
        g = goa.explode(index_parts=False).reset_index(drop=True)
        xy, line_of, is_end, seg_a, seg_b = [], [], [], [], []
        for i, geom in enumerate(g.geometry):
            c = np.asarray(geom.coords)[:, :2]
            base = len(xy)
            xy.extend(c)
            line_of.extend([g.OBJECTID.iat[i]] * len(c))
            is_end.extend([True] + [False] * (len(c) - 2) + [True] if len(c) > 1 else [True])
            seg_a.extend(range(base, base + len(c) - 1))
            seg_b.extend(range(base + 1, base + len(c)))
        self.xy = np.asarray(xy)
        self.line_of = np.asarray(line_of)
        is_end = np.asarray(is_end)
        n = len(self.xy)
        tree = cKDTree(self.xy)
        uf = _UF(n)
        ends = np.flatnonzero(is_end)
        for e, nbrs in zip(ends, tree.query_ball_point(self.xy[ends], stitch_tol_m)):
            for j in nbrs:
                uf.union(e, j)
        root = np.array([uf.find(i) for i in range(n)])
        _, self.node = np.unique(root, return_inverse=True)   # vertex -> graph node
        m = self.node.max() + 1
        a, b = self.node[seg_a], self.node[seg_b]
        w = np.linalg.norm(self.xy[seg_a] - self.xy[seg_b], axis=1)
        keep = a != b
        self.seg_a = np.asarray(seg_a)[keep]
        self.seg_b = np.asarray(seg_b)[keep]
        A = coo_matrix((np.r_[w[keep], w[keep]], (np.r_[a[keep], b[keep]], np.r_[b[keep], a[keep]])),
                       shape=(m, m)).tocsr()
        A.sum_duplicates()
        self.A = A
        # representative coordinate per graph node and edge -> GoA line id
        self.node_xy = np.zeros((m, 2))
        self.node_xy[self.node] = self.xy
        self.edge_line = {}
        for sa, sb in zip(self.seg_a, self.seg_b):
            self.edge_line[(self.node[sa], self.node[sb])] = self.line_of[sa]
            self.edge_line[(self.node[sb], self.node[sa])] = self.line_of[sa]
        self.tree = cKDTree(self.node_xy)
        self.n_stitched = int(n - m)
        log.info("GoA graph: %d vertices -> %d nodes, %d edges", n, m, A.nnz // 2)

    def snap(self, xy: np.ndarray, max_m: float):
        d, idx = self.tree.query(xy)
        idx = np.where(d <= max_m, idx, -1)
        return idx, d


def _augment(gg: GoAGraph, node_xy: dict, radius_m: float, fallback_m: float):
    """Add one virtual node per substation, joined to every GoA graph node
    within radius_m (several lines usually terminate around a substation
    without touching each other). Falls back to the single nearest node
    within fallback_m."""
    from scipy.sparse import bmat, csr_matrix
    codes = sorted(node_xy)
    xy = np.array([node_xy[c] for c in codes])
    m, k = gg.A.shape[0], len(codes)
    rows, cols, w, snap_d = [], [], [], {}
    for i, (c, nb) in enumerate(zip(codes, gg.tree.query_ball_point(xy, radius_m))):
        if not nb:
            d, j = gg.tree.query(xy[i])
            nb = [j] if d <= fallback_m else []
        ds = np.linalg.norm(gg.node_xy[nb] - xy[i], axis=1) if nb else np.array([])
        snap_d[c] = float(ds.min()) if len(ds) else np.inf
        rows += [i] * len(nb); cols += list(nb); w += list(np.maximum(ds, 1.0))
    B = csr_matrix((w, (rows, cols)), shape=(k, m))
    A = bmat([[gg.A, B.T], [B, None]]).tocsr()
    return A, {c: m + i for i, c in enumerate(codes)}, snap_d


def match_branches(gg: GoAGraph, branches, node_xy: dict, cfg):
    """branches: DataFrame with from_node, to_node, length_km.
    Returns ({branch index: {ratio, geometry, goa_lines}}, snap distances)."""
    t = cfg["topology"]
    lo, hi = t["match_ratio"]
    dlo, dhi = t["detour_ratio"]
    A, vnode, snap_d = _augment(gg, node_xy, t["substation_star_m"], t["substation_snap_m"])
    m = gg.A.shape[0]
    out, by_src = {}, {}
    for i, r in branches.iterrows():
        by_src.setdefault(r.from_node, []).append(i)
    for s, idxs in by_src.items():
        sv = vnode[s]
        limit = max(branches.loc[idxs, "length_km"].max() * 1000 * hi * 1.2, 5000)
        dist_m, pred = dijkstra(A, indices=sv, limit=limit, return_predecessors=True)
        for i in idxs:
            r = branches.loc[i]
            tv = vnode[r.to_node]
            if not np.isfinite(dist_m[tv]):
                out[i] = {"ratio": np.nan, "reason": "no_path"}
                continue
            route = r.length_km * 1000
            ratio = dist_m[tv] / route if route > 0 else np.inf
            crow = float(np.hypot(*(np.subtract(node_xy[s], node_xy[r.to_node]))))
            detour = dist_m[tv] / max(crow, 1.0)
            if lo <= ratio <= hi:
                how = "derived:goa_path_match"
            elif crow > 500 and dlo <= detour <= dhi:
                # path consistent with substation locations but not with the
                # CODERS segment length -> geometry kept, length flagged
                how = "derived:goa_path_match_crowfly"
            else:
                out[i] = {"ratio": ratio, "detour": detour, "reason": "length_mismatch"}
                continue
            path = [tv]
            while path[-1] != sv:
                path.append(pred[path[-1]])
            path = path[::-1]
            lines = {gg.edge_line.get((a, b)) for a, b in zip(path[:-1], path[1:])}
            coords = [node_xy[s]] + [tuple(gg.node_xy[p]) for p in path if p < m] + [node_xy[r.to_node]]
            out[i] = {"ratio": ratio, "detour": detour, "how": how,
                      "path_km": dist_m[tv] / 1000, "geometry": LineString(coords),
                      "goa_lines": sorted(x for x in lines if x is not None)}
    return out, snap_d
