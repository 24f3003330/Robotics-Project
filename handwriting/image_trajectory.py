"""
handwriting/image_trajectory.py - ESTIMATE a pen trajectory from a static
handwriting image, so image examples can enter the trajectory pipeline.

A static image records only where ink is. It does NOT record timing, speed,
pressure, pen tilt, stroke order, writing direction or pen lifts. This module
reconstructs a plausible drawing order from the ink shape; everything except
the ink geometry is an estimate:

    image -> grayscale -> Otsu binarisation (ink = dark) -> small-noise removal
          -> Zhang-Suen thinning (1-pixel skeleton)
          -> skeleton graph: segments between end points / junctions,
             short spurs (thinning artefacts) pruned
          -> strokes: per connected ink component, segments chained by
             smoothest continuation at junctions; a pen lift where the chain
             cannot continue
          -> components ordered in reading order (lines top to bottom,
             left to right within a line)

The result is a Trajectory with
    x, y       skeleton coordinates in pixels (from the ink: real geometry)
    pen        1 on reconstructed strokes, 0 at one lift point per stroke (ESTIMATED)
    t          a SEQUENCE INDEX (0.01 per point), not a time - there is no timing
    pressure, altitude, azimuth   NaN (unavailable)
Models trained on it must use features that ignore timing and pen channels
(config "geometric" feature set, arc-length resampling).
"""

from collections import defaultdict

import numpy as np

from .preprocessing import Trajectory

RECONSTRUCTED = ["stroke order", "writing direction within strokes", "pen-up / pen-down (stroke segmentation)"]
UNAVAILABLE = ["timestamps / timing", "speed and acceleration", "pressure", "pen altitude", "pen azimuth",
               "in-air (hover) movement"]
SEQUENCE_STEP = 0.01          # "t" increment per point: an ordering index, not seconds
MAX_SIDE = 2000               # larger images are downscaled first (the pipeline is scale-free)

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


# ---------------------------------------------------------------------------
# Image -> binary ink mask -> skeleton
# ---------------------------------------------------------------------------
# Formats accepted from browser uploads (sniffed from the bytes, not the file name).
UPLOAD_FORMATS = {"png": b"\x89PNG\r\n\x1a\n", "jpeg": b"\xff\xd8\xff"}


def sniff_upload_format(data):
    """'png' / 'jpeg' / 'webp' from the file's magic bytes, else None."""
    for name, magic in UPLOAD_FORMATS.items():
        if data.startswith(magic):
            return name
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def load_ink_mask(path):
    """-> (bool ink mask, info). Transparent pixels count as white paper."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"{path}: not a readable image")
    return ink_mask_from_image(img, path)


def ink_mask_from_image(img, name="image"):
    """Decoded image array (any channels) -> (bool ink mask, info)."""
    import cv2

    if img.ndim == 3 and img.shape[2] == 4:                      # alpha -> composite on white
        alpha = img[:, :, 3:4].astype(np.float64) / 255.0
        img = (img[:, :, :3] * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    scale = min(1.0, MAX_SIDE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inverted = False
    if ink.mean() > 0.5:                                         # light ink on dark paper
        ink, inverted = 1 - ink, True
    if not ink.any():
        raise ValueError(f"{name}: no ink found")
    # drop specks smaller than a fraction of a pen dot
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    width = max(1.0, 4.0 * float(np.median(dist[ink > 0])))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < max(4.0, 0.25 * width * width))
    ink[np.isin(labels, small[small > 0])] = 0
    if not ink.any():
        raise ValueError(f"{name}: only specks of ink found")
    return ink.astype(bool), {"image_size": [int(img.shape[1]), int(img.shape[0])],
                              "downscale": round(scale, 4), "inverted_polarity": inverted}


def thin(mask):
    """Zhang-Suen thinning -> 1-pixel-wide skeleton (bool array)."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            c = img[1:-1, 1:-1]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b = sum(p.astype(int) for p in ring[:8])
            a = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(int) for i in range(8))
            if step == 0:
                m1, m2 = p2 * p4 * p6, p4 * p6 * p8
            else:
                m1, m2 = p2 * p4 * p8, p2 * p6 * p8
            remove = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & (m1 == 0) & (m2 == 0)
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


# ---------------------------------------------------------------------------
# Skeleton -> segments -> strokes
# ---------------------------------------------------------------------------
def _adjacency(pix):
    """8-neighbourhood, but a diagonal neighbour only when no shared orthogonal
    neighbour exists - avoids fake junctions on staircase diagonals."""
    adj = {}
    for r, c in pix:
        out = [(r + dr, c + dc) for dr, dc in _ORTH if (r + dr, c + dc) in pix]
        out += [(r + dr, c + dc) for dr, dc in _DIAG
                if (r + dr, c + dc) in pix and (r + dr, c) not in pix and (r, c + dc) not in pix]
        adj[(r, c)] = out
    return adj


def _segments(adj):
    """-> list of pixel paths between nodes (end points / junctions), closed loops, isolated dots."""
    nodes = {p for p, n in adj.items() if len(n) != 2}
    seen_edges, on_path, segs = set(), set(), []
    for p in sorted(nodes):
        if not adj[p]:
            segs.append([p])
            on_path.add(p)
            continue
        for q in adj[p]:
            if (p, q) in seen_edges:
                continue
            path, prev, cur = [p, q], p, q
            seen_edges |= {(p, q), (q, p)}
            while cur not in nodes:
                nxt = [n for n in adj[cur] if n != prev and (cur, n) not in seen_edges]
                if not nxt:
                    break
                seen_edges |= {(cur, nxt[0]), (nxt[0], cur)}
                path.append(nxt[0])
                prev, cur = cur, nxt[0]
            segs.append(path)
            on_path.update(path)
    for p in sorted(adj):                                   # loops made only of degree-2 pixels
        if p in on_path:
            continue
        path, prev, cur = [p], None, p
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in on_path and n not in path]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        on_path.update(path)
        if len(path) > 2:
            i = min(range(len(path)), key=lambda k: (path[k][1], path[k][0]))   # start leftmost
            path = path[i:] + path[:i]
            path.append(path[0])
        segs.append(path)
    return segs, nodes


def _direction(points, from_start=True, n=6):
    pts = np.asarray(points[:n] if from_start else points[-n:], dtype=float)
    if len(pts) < 2:
        return None
    d = pts[-1] - pts[0]
    norm = np.hypot(*d)
    return d / norm if norm > 0 else None


def trace_strokes(skeleton, stroke_width):
    """Skeleton -> list of strokes, each an (N, 2) array of (x, y), in estimated writing order."""
    pix = set(zip(*np.nonzero(skeleton)))
    adj = _adjacency(pix)
    segs, nodes = _segments(adj)
    # node id per end pixel: adjacent junction pixels form one junction
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            a = parent[a]
        return a

    for p in nodes:
        if len(adj[p]) >= 3:
            for q in adj[p]:
                if q in nodes and len(adj[q]) >= 3:
                    parent[find(p)] = find(q)
    node = lambda p: ("n", find(p)) if p in nodes else ("p", p)
    ends = [(node(s[0]), node(s[-1])) for s in segs]
    # drop junction-internal links and short spurs (thinning artefacts)
    degree = defaultdict(int)
    for a, b in ends:
        degree[a] += 1
        degree[b] += 1
    spur = max(3, int(round(2 * stroke_width)))
    keep = []
    for i, (s, (a, b)) in enumerate(zip(segs, ends)):
        if len(s) > 1 and a == b and len(s) <= 3:
            continue
        if len(s) < spur and len(s) > 1 and min(degree[a], degree[b]) == 1 and max(degree[a], degree[b]) >= 3:
            continue
        keep.append(i)
    segs, ends = [segs[i] for i in keep], [ends[i] for i in keep]
    if not segs:
        return []
    # connected components of the segment graph
    comp_parent = list(range(len(segs)))

    def cfind(i):
        while comp_parent[i] != i:
            comp_parent[i] = comp_parent[comp_parent[i]]
            i = comp_parent[i]
        return i

    by_node = defaultdict(list)
    for i, (a, b) in enumerate(ends):
        by_node[a].append((i, 0))
        by_node[b].append((i, 1))
    for items in by_node.values():
        for i, _ in items[1:]:
            comp_parent[cfind(i)] = cfind(items[0][0])
    comps = defaultdict(list)
    for i in range(len(segs)):
        comps[cfind(i)].append(i)
    comps = list(comps.values())
    # reading order: a component joins the text line its vertical extent overlaps
    # most (>= half its own height; ascenders / descenders still overlap the
    # line's body), lines top to bottom, components left to right within a line
    boxes = []
    for c in comps:
        rc = np.array([p for i in c for p in segs[i]])
        boxes.append((rc[:, 0].min(), rc[:, 0].max(), rc[:, 1].min()))
    order = sorted(range(len(comps)), key=lambda k: (boxes[k][0] + boxes[k][1]) / 2)
    lines = []                                        # [top, bottom, members]
    for k in order:
        top, bottom = boxes[k][0], boxes[k][1]
        height = bottom - top + 1
        best, best_overlap = None, 0.0
        for line in lines:
            overlap = min(bottom, line[1]) - max(top, line[0]) + 1
            if overlap > best_overlap:
                best, best_overlap = line, overlap
        if best is not None and best_overlap >= 0.5 * min(height, best[1] - best[0] + 1):
            best[0], best[1] = min(best[0], top), max(best[1], bottom)
            best[2].append(k)
        else:
            lines.append([top, bottom, [k]])
    lines.sort(key=lambda line: np.median([(boxes[j][0] + boxes[j][1]) / 2 for j in line[2]]))
    ordered = [k for line in lines for k in sorted(line[2], key=lambda k: boxes[k][2])]

    strokes = []
    for k in ordered:
        unused = set(comps[k])
        while unused:
            # start at the leftmost dead end if there is one, else at the leftmost segment start
            open_nodes = [n for n, items in by_node.items()
                          if sum(i in unused for i, _ in items) == 1]
            if open_nodes:
                start = min(open_nodes, key=lambda n: min(
                    (segs[i][0] if e == 0 else segs[i][-1])[1] for i, e in by_node[n] if i in unused))
            else:
                i0 = min(unused, key=lambda i: segs[i][0][1])
                start = ends[i0][0]
            cur, heading, pts = start, np.array([0.0, 1.0]), []      # (row, col): initially rightwards
            while True:
                options = [(i, e) for i, e in by_node[cur] if i in unused]
                if not options:
                    break

                def turn(opt):
                    i, e = opt
                    path = segs[i] if e == 0 else segs[i][::-1]
                    d = _direction(path)
                    return 2.0 if d is None else 1.0 - float(np.dot(d, heading))

                i, e = min(options, key=turn)
                path = segs[i] if e == 0 else segs[i][::-1]
                pts.extend(path if not pts else path[1:])
                unused.discard(i)
                d = _direction(path, from_start=False)
                heading = heading if d is None else d
                cur = ends[i][1] if e == 0 else ends[i][0]
            if pts:
                strokes.append(np.array([(c, r) for r, c in pts], dtype=float))   # -> (x, y)
    return strokes


def image_to_trajectory(path):
    """-> (Trajectory, info). See the module docstring for what is estimated."""
    ink, info = load_ink_mask(path)
    return _trajectory_from_ink(ink, info, path)


def image_bytes_to_trajectory(data, name="uploaded image"):
    """Encoded image bytes (e.g. a browser upload) -> (Trajectory, info), through exactly
    the same reconstruction as image files. Raises ValueError for unusable images."""
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"{name}: the image could not be decoded (corrupt or unsupported)")
    ink, info = ink_mask_from_image(img, name)
    return _trajectory_from_ink(ink, info, name)


def _trajectory_from_ink(ink, info, name):
    skeleton = thin(ink)
    width = float(ink.sum()) / max(1, int(skeleton.sum()))
    strokes = trace_strokes(skeleton, width)
    if not strokes:
        raise ValueError(f"{name}: no strokes could be traced")
    xs, ys, pen = [], [], []
    for s in strokes:
        xs += list(s[:, 0]) + [s[-1, 0]]
        ys += list(s[:, 1]) + [s[-1, 1]]
        pen += [1.0] * len(s) + [0.0]                     # one estimated pen lift per stroke
    t = np.arange(len(xs)) * SEQUENCE_STEP
    info.update({"method": "binarise + Zhang-Suen skeleton + stroke tracing",
                 "strokes": len(strokes), "stroke_width_px": round(width, 2), "points": len(xs),
                 "reconstructed": RECONSTRUCTED, "unavailable": UNAVAILABLE,
                 "t_meaning": "sequence index (not time)"})
    return Trajectory(x=xs, y=ys, t=t, pen=pen), info
